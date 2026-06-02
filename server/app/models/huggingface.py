from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from huggingface_hub import snapshot_download

from app.config import Settings
from app.schemas import CaptionPrediction, WordCaption

from .interface import FrameForInference


class HuggingFaceKslModel:
    name = "huggingface"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ready = False
        self._module: ModuleType | None = None
        self._model: Any = None
        self._predict_fn: Callable[..., Any] | None = None
        self._predict_accepts_model = True
        self._word_classifier: _WordClassifierRuntime | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    async def load(self) -> None:
        if not self._settings.hf_model_id:
            raise ValueError("HF_MODEL_ID is required when MODEL_BACKEND=huggingface.")

        model_dir = await asyncio.to_thread(self._resolve_model_dir)
        inference_file = model_dir / "inference.py"
        if inference_file.exists():
            module = await asyncio.to_thread(_load_inference_module, model_dir)

            load_model = getattr(module, "load_model", None)
            predict = getattr(module, "predict", None)
            if not callable(load_model):
                raise ValueError("Model repo inference.py must define load_model(model_dir, device).")
            if not callable(predict):
                raise ValueError("Model repo inference.py must define predict(...).")

            self._module = module
            self._predict_fn = predict
            self._predict_accepts_model = _predict_accepts_model(predict)
            self._model = await asyncio.to_thread(load_model, str(model_dir), self._settings.model_device)
        elif (model_dir / "best.pt").exists() and (model_dir / "model" / "predict_word_classifier.py").exists():
            self._word_classifier = await asyncio.to_thread(
                _WordClassifierRuntime.load,
                model_dir,
                self._settings,
            )
        else:
            raise ValueError(
                "Model repo must include either inference.py or best.pt with model/predict_word_classifier.py."
            )
        self._ready = True

    async def close(self) -> None:
        close_model = getattr(self._model, "close", None)
        if callable(close_model):
            await asyncio.to_thread(close_model)
        if self._word_classifier is not None:
            await asyncio.to_thread(self._word_classifier.close)
        self._ready = False
        self._module = None
        self._model = None
        self._predict_fn = None
        self._word_classifier = None

    async def predict(self, frames: list[FrameForInference]) -> list[CaptionPrediction]:
        if not self._ready or self._predict_fn is None:
            if self._word_classifier is None:
                raise RuntimeError("Hugging Face KSL model is not loaded.")
        if not frames:
            return []

        if self._word_classifier is not None:
            return await asyncio.to_thread(self._word_classifier.predict, frames)

        raw_predictions = await asyncio.to_thread(self._predict_sync, frames)
        return _normalize_predictions(raw_predictions, frames)

    def _resolve_model_dir(self) -> Path:
        model_id = self._settings.hf_model_id
        if model_id is None:
            raise ValueError("HF_MODEL_ID is required.")

        local_path = Path(model_id).expanduser()
        if local_path.exists():
            return local_path.resolve()

        download_kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "revision": self._settings.hf_model_revision,
        }
        if self._settings.hf_token:
            download_kwargs["token"] = self._settings.hf_token
        return Path(snapshot_download(**download_kwargs))

    def _predict_sync(self, frames: list[FrameForInference]) -> Any:
        if self._predict_fn is None:
            raise RuntimeError("Model predict function is not loaded.")

        images = [frame.image_rgb for frame in frames]
        timestamps_ms = [frame.timestamp_ms for frame in frames]
        if self._predict_accepts_model:
            return self._predict_fn(self._model, images, timestamps_ms)
        return self._predict_fn(images, timestamps_ms)


class _WordClassifierRuntime:
    def __init__(
        self,
        *,
        module: ModuleType,
        model: Any,
        label_to_id: dict[str, int],
        checkpoint: dict[str, Any],
        extractor: Any,
        device: Any,
        max_frames: int,
        target_fps: float,
        top_k: int,
    ) -> None:
        self._module = module
        self._model = model
        self._label_to_id = label_to_id
        self._checkpoint = checkpoint
        self._extractor = extractor
        self._device = device
        self._max_frames = max_frames
        self._target_fps = target_fps
        self._top_k = top_k

    @classmethod
    def load(cls, model_dir: Path, settings: Settings) -> "_WordClassifierRuntime":
        module = _import_repo_module(model_dir, "model.predict_word_classifier")
        device = _select_torch_device(settings.model_device)
        checkpoint_path = model_dir / "best.pt"
        model, label_to_id, checkpoint = module.load_model(checkpoint_path, device)
        max_frames = int(checkpoint["config"].get("max_frames", getattr(model.config, "max_frames", 64)))
        task_model_path = model_dir / "model" / "assets" / "holistic_landmarker.task"
        extractor = module.HolisticKeypointExtractor(
            module.HolisticExtractorConfig(
                target_fps=settings.sequence_target_fps,
                task_model_path=str(task_model_path),
                max_frames=max_frames,
            )
        )
        return cls(
            module=module,
            model=model,
            label_to_id=label_to_id,
            checkpoint=checkpoint,
            extractor=extractor,
            device=device,
            max_frames=max_frames,
            target_fps=settings.sequence_target_fps,
            top_k=settings.model_top_k,
        )

    def close(self) -> None:
        close_extractor = getattr(self._extractor, "close", None)
        if callable(close_extractor):
            close_extractor()

    def predict(self, frames: list[FrameForInference]) -> list[CaptionPrediction]:
        if not frames:
            return []

        with tempfile.TemporaryDirectory(prefix="ksl_segment_") as tmp_dir:
            video_path = _write_video_clip(Path(tmp_dir), frames, self._target_fps)
            keypoints = self._extractor.extract_video(video_path)

        result = self._module.predict(
            model=self._model,
            keypoints=keypoints,
            label_to_id=self._label_to_id,
            max_frames=self._max_frames,
            device=self._device,
            top_k=self._top_k,
        )
        raw_prediction = result["prediction"]
        start_ms = int(frames[0].timestamp_ms or 0)
        end_ms = int(frames[-1].timestamp_ms or start_ms)
        if end_ms <= start_ms:
            end_ms = start_ms + int((len(frames) / self._target_fps) * 1000)

        text = str(raw_prediction["label"])
        confidence = float(raw_prediction["probability"])
        return [
            CaptionPrediction(
                frame_id=frames[-1].frame_id,
                text=text,
                words=[
                    WordCaption(
                        text=text,
                        confidence=confidence,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                ],
                is_final=True,
            )
        ]


def _select_torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"MODEL_DEVICE={device_name} was requested, but CUDA is not available.")
    return torch.device(device_name)


def _write_video_clip(output_dir: Path, frames: list[FrameForInference], fps: float) -> Path:
    import cv2
    import numpy as np

    first_frame = frames[0].image_rgb
    height, width = first_frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input frames must have positive width and height.")

    for suffix, codec in ((".mp4", "mp4v"), (".avi", "MJPG")):
        output_path = output_dir / f"segment{suffix}"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            continue
        try:
            for frame in frames:
                image_rgb = frame.image_rgb
                if image_rgb.shape[:2] != (height, width):
                    image_rgb = cv2.resize(image_rgb, (width, height))
                bgr_frame = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
        finally:
            writer.release()
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

    raise RuntimeError("Could not create a temporary video clip for model inference.")


def _import_repo_module(model_dir: Path, module_name: str) -> ModuleType:
    model_path = str(model_dir)
    if model_path in sys.path:
        sys.path.remove(model_path)
    sys.path.insert(0, model_path)
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    return importlib.import_module(module_name)


def _load_inference_module(model_dir: Path) -> ModuleType:
    inference_file = model_dir / "inference.py"
    if not inference_file.exists():
        raise ValueError(f"Model repo is missing inference.py: {inference_file}")

    model_path = str(model_dir)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    module_name = f"_ksl_hf_inference_{abs(hash(str(inference_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, inference_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load inference.py from {inference_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _predict_accepts_model(predict_fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return True

    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values()):
        return True
    return len(positional_params) >= 3


def _normalize_predictions(raw_predictions: Any, frames: list[FrameForInference]) -> list[CaptionPrediction]:
    if not isinstance(raw_predictions, list):
        raise ValueError("Model predict(...) must return a list.")
    if len(raw_predictions) != len(frames):
        raise ValueError("Model predict(...) must return one prediction per input frame.")

    predictions: list[CaptionPrediction] = []
    for raw_prediction, frame in zip(raw_predictions, frames, strict=True):
        if isinstance(raw_prediction, CaptionPrediction):
            prediction = raw_prediction.model_copy(update={"frame_id": frame.frame_id})
            predictions.append(prediction)
            continue

        if not isinstance(raw_prediction, dict):
            raise ValueError("Each model prediction must be a dict.")

        start_ms = int(frame.timestamp_ms or 0)
        words = _normalize_words(raw_prediction.get("words", []), start_ms)
        text = str(raw_prediction.get("text", ""))
        if not text and words:
            text = " ".join(word.text for word in words if word.text)

        predictions.append(
            CaptionPrediction(
                frame_id=frame.frame_id,
                text=text,
                words=words,
                is_final=bool(raw_prediction.get("is_final", True)),
            )
        )
    return predictions


def _normalize_words(raw_words: Any, fallback_start_ms: int) -> list[WordCaption]:
    if raw_words is None:
        return []
    if not isinstance(raw_words, list):
        raise ValueError("Prediction words must be a list.")

    words: list[WordCaption] = []
    for raw_word in raw_words:
        if isinstance(raw_word, WordCaption):
            words.append(raw_word)
            continue
        if not isinstance(raw_word, dict):
            raise ValueError("Each prediction word must be a dict.")

        start_ms = int(raw_word.get("start_ms", fallback_start_ms) or 0)
        end_ms = int(raw_word.get("end_ms", start_ms) or start_ms)
        if end_ms < start_ms:
            end_ms = start_ms

        words.append(
            WordCaption(
                text=str(raw_word.get("text", "")),
                confidence=float(raw_word.get("confidence", 0.0)),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return words
