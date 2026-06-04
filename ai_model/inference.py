from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


class EmptyKslRecognizer:
    """Conservative fallback used when no trained MediaPipe artifact is present."""

    def predict_one(self, frame: Any, timestamp_ms: int | None) -> dict:
        return {"text": "", "words": [], "is_final": False}


def load_model(model_dir: str, device: str) -> Any:
    """Load the MediaPipe MVP KSL word recognition pipeline."""

    root = Path(model_dir)
    if (root / "mediapipe_mvp.joblib").exists():
        from mediapipe_mvp import load_mediapipe_mvp

        return load_mediapipe_mvp(root, device)

    return EmptyKslRecognizer()


def predict(model: Any, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    """Return one word-level caption prediction per input RGB frame."""

    if len(frames) != len(timestamps_ms):
        raise ValueError("frames and timestamps_ms must have the same length.")
    return [model.predict_one(frame, timestamp_ms) for frame, timestamp_ms in zip(frames, timestamps_ms, strict=True)]
