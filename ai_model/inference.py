from __future__ import annotations

import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from modeling import KslWordRecognizer, load_json, load_labels


def load_model(model_dir: str, device: str) -> KslWordRecognizer:
    """Load the YOLO-assisted VideoMAE KSL word recognition pipeline."""

    root = Path(model_dir)
    labels = load_labels(root / "labels.json")
    preprocessor = load_json(root / "preprocessor_config.json", default={})
    config = load_json(root / "model_config.json", default={})
    return KslWordRecognizer(
        model_dir=root,
        labels=labels,
        preprocessor=preprocessor,
        config=config,
        device=device,
    )


def predict(model: KslWordRecognizer, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    """Return one word-level caption prediction per input RGB frame."""

    if len(frames) != len(timestamps_ms):
        raise ValueError("frames and timestamps_ms must have the same length.")
    return [model.predict_one(frame, timestamp_ms) for frame, timestamp_ms in zip(frames, timestamps_ms, strict=True)]
