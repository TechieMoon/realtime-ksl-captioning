from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
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

    @property
    def ready(self) -> bool:
        return self._ready

    async def load(self) -> None:
        if not self._settings.hf_model_id:
            raise ValueError("HF_MODEL_ID is required when MODEL_BACKEND=huggingface.")

        model_dir = await asyncio.to_thread(self._resolve_model_dir)
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
        self._ready = True

    async def close(self) -> None:
        close_model = getattr(self._model, "close", None)
        if callable(close_model):
            await asyncio.to_thread(close_model)
        self._ready = False
        self._module = None
        self._model = None
        self._predict_fn = None

    async def predict(self, frames: list[FrameForInference]) -> list[CaptionPrediction]:
        if not self._ready or self._predict_fn is None:
            raise RuntimeError("Hugging Face KSL model is not loaded.")
        if not frames:
            return []

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

        start_ms = frame.timestamp_ms or 0
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
