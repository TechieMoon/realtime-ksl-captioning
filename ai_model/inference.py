from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Label:
    id: int
    text: str
    name: str


class KslWordRecognizer:
    """Small real-time inference wrapper ready for trained weights later."""

    def __init__(
        self,
        *,
        labels: list[Label],
        preprocessor: dict[str, Any],
        config: dict[str, Any],
        device: str,
        prototype_features: np.ndarray | None = None,
        prototype_label_ids: np.ndarray | None = None,
    ) -> None:
        self.labels = {label.id: label for label in labels}
        self.preprocessor = preprocessor
        self.config = config
        self.device = device
        self.prototype_features = prototype_features
        self.prototype_label_ids = prototype_label_ids
        smoothing = config.get("smoothing", {})
        self.history: deque[tuple[int, float]] = deque(maxlen=int(smoothing.get("window_size", 5)))
        self.previous_frame: np.ndarray | None = None

    def predict_one(self, frame: np.ndarray, timestamp_ms: int | None) -> dict[str, Any]:
        rgb = _validate_rgb_frame(frame)
        features = _extract_features(rgb, self.previous_frame)
        self.previous_frame = _downsample_for_motion(rgb)

        label_id, confidence = self._classify(features)
        label_id, confidence, is_final = self._smooth(label_id, confidence)
        label = self.labels.get(label_id, self.labels[0])

        start_ms = int(timestamp_ms or 0)
        duration_ms = int(self.config.get("default_word_duration_ms", 500))
        min_confidence = float(self.config.get("min_emit_confidence", 0.55))

        if label.id == 0 or confidence < min_confidence:
            return {"text": "", "words": [], "is_final": False}

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

    def _classify(self, features: np.ndarray) -> tuple[int, float]:
        if self.prototype_features is not None and self.prototype_label_ids is not None:
            return _nearest_prototype(features, self.prototype_features, self.prototype_label_ids)

        # Training has not happened yet. Keep live output conservative so unknown
        # signs do not become confident captions during integration tests.
        motion_energy = float(features[0])
        min_motion = float(self.config.get("prototype_min_motion", 0.08))
        emit_demo_label = bool(self.config.get("emit_demo_label_without_weights", False))
        if not emit_demo_label or motion_energy < min_motion:
            return 0, 0.0

        demo_label_id = int(self.config.get("demo_label_id", 1))
        confidence = min(0.35, 0.15 + motion_energy)
        return demo_label_id, confidence

    def _smooth(self, label_id: int, confidence: float) -> tuple[int, float, bool]:
        self.history.append((label_id, confidence))
        if not self.history:
            return label_id, confidence, False

        votes: dict[int, list[float]] = {}
        for past_label_id, past_confidence in self.history:
            votes.setdefault(past_label_id, []).append(past_confidence)

        best_label_id, confidences = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1])))
        mean_confidence = float(sum(confidences) / len(confidences))
        stable_frames = int(self.config.get("smoothing", {}).get("stable_frames", 3))
        is_final = best_label_id != 0 and len(confidences) >= stable_frames
        return best_label_id, mean_confidence, is_final


def load_model(model_dir: str, device: str) -> KslWordRecognizer:
    """Load labels, preprocessing config, and optional prototype weights."""

    root = Path(model_dir)
    labels = _load_labels(root / "labels.json")
    preprocessor = _load_json(root / "preprocessor_config.json", default={})
    config = _load_json(root / "model_config.json", default={})

    prototype_features: np.ndarray | None = None
    prototype_label_ids: np.ndarray | None = None
    prototype_path = root / "prototype_features.npz"
    if prototype_path.exists():
        archive = np.load(prototype_path)
        prototype_features = archive["features"].astype(np.float32)
        prototype_label_ids = archive["label_ids"].astype(np.int64)

    return KslWordRecognizer(
        labels=labels,
        preprocessor=preprocessor,
        config=config,
        device=device,
        prototype_features=prototype_features,
        prototype_label_ids=prototype_label_ids,
    )


def predict(model: KslWordRecognizer, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    """Return one caption prediction per RGB frame."""

    if len(frames) != len(timestamps_ms):
        raise ValueError("frames and timestamps_ms must have the same length.")
    return [model.predict_one(frame, timestamp_ms) for frame, timestamp_ms in zip(frames, timestamps_ms, strict=True)]


def _load_labels(path: Path) -> list[Label]:
    data = _load_json(path, default=None)
    if data is None:
        raise FileNotFoundError(f"Missing labels file: {path}")

    raw_labels = data.get("labels", data)
    labels = [Label(id=int(item["id"]), text=str(item["text"]), name=str(item.get("name", item["id"]))) for item in raw_labels]
    if not any(label.id == 0 for label in labels):
        labels.insert(0, Label(id=0, text="", name="unknown"))
    return labels


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Each frame must be a numpy.ndarray.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Each frame must have shape (height, width, 3).")
    if frame.dtype != np.uint8:
        raise ValueError("Each frame must use dtype uint8.")
    return frame


def _extract_features(frame: np.ndarray, previous_frame: np.ndarray | None) -> np.ndarray:
    small = _downsample_for_motion(frame)
    if previous_frame is None:
        motion_energy = 0.0
        center_motion = 0.0
    else:
        diff = np.abs(small.astype(np.float32) - previous_frame.astype(np.float32)) / 255.0
        motion_energy = float(diff.mean())
        h, w, _ = diff.shape
        center = diff[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
        center_motion = float(center.mean()) if center.size else motion_energy

    channels = frame.astype(np.float32) / 255.0
    red_mean = float(channels[:, :, 0].mean())
    green_mean = float(channels[:, :, 1].mean())
    blue_mean = float(channels[:, :, 2].mean())
    brightness = float(channels.mean())
    contrast = float(channels.std())
    skin_fraction = _estimate_skin_fraction(frame)

    return np.asarray(
        [
            motion_energy,
            center_motion,
            red_mean,
            green_mean,
            blue_mean,
            brightness,
            contrast,
            skin_fraction,
        ],
        dtype=np.float32,
    )


def _downsample_for_motion(frame: np.ndarray, size: int = 64) -> np.ndarray:
    height, width, _ = frame.shape
    y_idx = np.linspace(0, height - 1, num=min(size, height)).astype(np.int64)
    x_idx = np.linspace(0, width - 1, num=min(size, width)).astype(np.int64)
    return frame[np.ix_(y_idx, x_idx)]


def _estimate_skin_fraction(frame: np.ndarray) -> float:
    rgb = frame.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    mask = (r > 95) & (g > 40) & (b > 20) & ((r - g) > 15) & (r > b)
    return float(mask.mean())


def _nearest_prototype(
    features: np.ndarray,
    prototype_features: np.ndarray,
    prototype_label_ids: np.ndarray,
) -> tuple[int, float]:
    distances = np.linalg.norm(prototype_features - features[None, :], axis=1)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    confidence = float(np.exp(-best_distance))
    return int(prototype_label_ids[best_index]), min(max(confidence, 0.0), 1.0)
