from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class Label:
    id: int
    text: str
    name: str


@dataclass(frozen=True, slots=True)
class RoiResult:
    image_rgb: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    label_id: int
    confidence: float
    ready: bool


class YoloRoiExtractor:
    """Optional YOLO-based signer ROI extractor with full-frame fallback."""

    def __init__(self, model_dir: Path, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.25))
        self.padding = float(config.get("padding", 0.15))
        self.class_ids = set(config.get("class_ids", []))
        self.model = None

        weights_path = config.get("weights_path")
        if weights_path:
            resolved_path = Path(weights_path)
            if not resolved_path.is_absolute():
                resolved_path = model_dir / resolved_path
            if resolved_path.exists():
                self.model = self._load_yolo(resolved_path)

    def extract(self, frame: np.ndarray) -> RoiResult:
        if self.enabled and self.model is not None:
            result = self._detect(frame)
            if result is not None:
                return result
        height, width, _ = frame.shape
        return RoiResult(
            image_rgb=frame,
            bbox_xyxy=(0, 0, width, height),
            confidence=0.0,
            source="full_frame",
        )

    def _load_yolo(self, weights_path: Path) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO weights were configured, but ultralytics is not installed. "
                "Install ai_model/requirements.txt before using YOLO ROI."
            ) from exc
        return YOLO(str(weights_path))

    def _detect(self, frame: np.ndarray) -> RoiResult | None:
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        if not results:
            return None

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        candidates: list[tuple[float, float, tuple[int, int, int, int]]] = []
        height, width, _ = frame.shape
        for box in boxes:
            class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
            if self.class_ids and class_id not in self.class_ids:
                continue
            confidence = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            x1, y1, x2, y2 = _pad_box(xyxy, width=width, height=height, padding=self.padding)
            area = max(1, (x2 - x1) * (y2 - y1))
            candidates.append((confidence * area, confidence, (x1, y1, x2, y2)))

        if not candidates:
            return None

        _, confidence, bbox = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return RoiResult(image_rgb=crop, bbox_xyxy=bbox, confidence=confidence, source="yolo")


class VideoMAEWordClassifier:
    """VideoMAE clip classifier that can be activated by adding a checkpoint."""

    def __init__(
        self,
        model_dir: Path,
        labels: list[Label],
        preprocessor: dict[str, Any],
        config: dict[str, Any],
        device: str,
    ) -> None:
        self.labels = labels
        self.preprocessor = preprocessor
        self.config = config
        self.device = device
        self.clip_size = int(preprocessor.get("clip_size", config.get("clip_size", 8)))
        self.target_size = int(preprocessor.get("target_size", 224))
        self.mean = np.asarray(preprocessor.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(preprocessor.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.clip_buffer: deque[np.ndarray] = deque(maxlen=self.clip_size)
        self.torch = None
        self.model = None

        checkpoint_path = _resolve_optional_path(model_dir, config.get("checkpoint_path"))
        if checkpoint_path is None:
            default_checkpoint = model_dir / "videomae"
            checkpoint_path = default_checkpoint if default_checkpoint.exists() else None
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    @property
    def has_checkpoint(self) -> bool:
        return self.model is not None

    def predict(self, roi_rgb: np.ndarray) -> ClassifierResult:
        self.clip_buffer.append(_resize_rgb(roi_rgb, self.target_size))
        if len(self.clip_buffer) < self.clip_size:
            return ClassifierResult(label_id=0, confidence=0.0, ready=False)
        if self.model is None:
            return ClassifierResult(label_id=0, confidence=0.0, ready=True)
        return self._predict_with_videomae(list(self.clip_buffer))

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            import torch
            from transformers import VideoMAEForVideoClassification
        except ImportError as exc:
            raise RuntimeError(
                "VideoMAE checkpoint was found, but torch/transformers are not installed. "
                "Install ai_model/requirements.txt before using the trained classifier."
            ) from exc

        self.torch = torch
        self.model = VideoMAEForVideoClassification.from_pretrained(str(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()

    def _predict_with_videomae(self, clip: list[np.ndarray]) -> ClassifierResult:
        assert self.torch is not None
        assert self.model is not None

        tensor = self.torch.from_numpy(_normalize_clip(clip, self.mean, self.std))
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(pixel_values=tensor)
            probabilities = self.torch.softmax(outputs.logits, dim=-1)[0]
            confidence, class_index = self.torch.max(probabilities, dim=-1)
        label_id = int(class_index.item())
        return ClassifierResult(label_id=label_id, confidence=float(confidence.item()), ready=True)


class KslWordRecognizer:
    """YOLO-assisted VideoMAE inference pipeline for word-level KSL captions."""

    def __init__(
        self,
        *,
        model_dir: Path,
        labels: list[Label],
        preprocessor: dict[str, Any],
        config: dict[str, Any],
        device: str,
    ) -> None:
        self.labels = {label.id: label for label in labels}
        self.config = config
        self.roi_extractor = YoloRoiExtractor(model_dir, config.get("roi_detector", {}))
        self.classifier = VideoMAEWordClassifier(
            model_dir=model_dir,
            labels=labels,
            preprocessor=preprocessor,
            config=config.get("classifier", {}),
            device=device,
        )
        smoothing = config.get("smoothing", {})
        self.history: deque[tuple[int, float]] = deque(maxlen=int(smoothing.get("window_size", 5)))

    def predict_one(self, frame: np.ndarray, timestamp_ms: int | None) -> dict[str, Any]:
        rgb = validate_rgb_frame(frame)
        roi = self.roi_extractor.extract(rgb)
        result = self.classifier.predict(roi.image_rgb)
        label_id, confidence, is_final = self._smooth(result.label_id, result.confidence)

        label = self.labels.get(label_id, self.labels[0])
        min_confidence = float(self.config.get("min_emit_confidence", 0.55))
        if not result.ready or label.id == 0 or confidence < min_confidence:
            return {"text": "", "words": [], "is_final": False}

        start_ms = int(timestamp_ms or 0)
        duration_ms = int(self.config.get("default_word_duration_ms", 500))
        return {
            "text": label.text,
            "words": [
                {
                    "text": label.text,
                    "confidence": round(float(confidence), 4),
                    "start_ms": start_ms,
                    "end_ms": start_ms + duration_ms,
                }
            ],
            "is_final": is_final,
        }

    def _smooth(self, label_id: int, confidence: float) -> tuple[int, float, bool]:
        self.history.append((label_id, confidence))
        votes: dict[int, list[float]] = {}
        for past_label_id, past_confidence in self.history:
            votes.setdefault(past_label_id, []).append(past_confidence)

        best_label_id, confidences = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1])))
        mean_confidence = float(sum(confidences) / len(confidences))
        stable_frames = int(self.config.get("smoothing", {}).get("stable_frames", 3))
        is_final = best_label_id != 0 and len(confidences) >= stable_frames
        return best_label_id, mean_confidence, is_final


def load_labels(path: Path) -> list[Label]:
    data = load_json(path, default=None)
    if data is None:
        raise FileNotFoundError(f"Missing labels file: {path}")

    raw_labels = data.get("labels", data)
    labels = [Label(id=int(item["id"]), text=str(item["text"]), name=str(item.get("name", item["id"]))) for item in raw_labels]
    if not any(label.id == 0 for label in labels):
        labels.insert(0, Label(id=0, text="", name="unknown"))
    return labels


def load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Each frame must be a numpy.ndarray.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Each frame must have shape (height, width, 3).")
    if frame.dtype != np.uint8:
        raise ValueError("Each frame must use dtype uint8.")
    return frame


def _resolve_optional_path(model_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = model_dir / path
    return path if path.exists() else None


def _resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(frame)
    image = image.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _normalize_clip(clip: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    frames = np.stack(clip).astype(np.float32) / 255.0
    frames = (frames - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return np.transpose(frames, (0, 3, 1, 2)).astype(np.float32)


def _pad_box(
    xyxy: np.ndarray,
    *,
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    x1 = max(0, int(x1 - box_width * padding))
    y1 = max(0, int(y1 - box_height * padding))
    x2 = min(width, int(x2 + box_width * padding))
    y2 = min(height, int(y2 + box_height * padding))
    return x1, y1, x2, y2
