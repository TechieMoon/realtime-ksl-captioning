from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_MODEL_DIR = REPO_ROOT / "ai_model"


def main() -> None:
    args = _parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    api = HfApi(token=token)
    model_card = _model_card_path(args.repo_id, generate=not args.no_generate_card)

    files = [
        (AI_MODEL_DIR / "mediapipe_mvp.joblib", "mediapipe_mvp.joblib"),
        (AI_MODEL_DIR / "metrics_full_mediapipe.json", "metrics_full_mediapipe.json"),
        (AI_MODEL_DIR / "metrics_mediapipe_mvp.json", "metrics_mediapipe_mvp.json"),
        (AI_MODEL_DIR / "labels.json", "labels.json"),
        (model_card, "README.md"),
        (AI_MODEL_DIR / "inference.py", "inference.py"),
        (AI_MODEL_DIR / "mediapipe_mvp.py", "mediapipe_mvp.py"),
        (AI_MODEL_DIR / "requirements.txt", "requirements.txt"),
    ]
    for local_path, repo_path in files:
        if not local_path.exists():
            print(f"skip missing: {local_path}")
            continue
        print(f"upload {local_path} -> {args.repo_id}/{repo_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    print(f"Uploaded model package to https://huggingface.co/{args.repo_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the trained MediaPipe model package to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Example: TechieMoon/realtime-ksl-captioning-mediapipe-mvp")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN.")
    parser.add_argument("--commit-message", default="Upload full MediaPipe KSL model")
    parser.add_argument("--no-generate-card", action="store_true")
    return parser.parse_args()


def _model_card_path(repo_id: str, *, generate: bool) -> Path:
    if not generate:
        return AI_MODEL_DIR / "README.md"

    metrics_path = AI_MODEL_DIR / "metrics_full_mediapipe.json"
    if not metrics_path.exists():
        return AI_MODEL_DIR / "README.md"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics.get("best", {})
    candidate = best.get("candidate", {})
    labels = metrics.get("labels", [])
    card_path = REPO_ROOT / "training" / "runs" / "hf_model_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        f"""---
library_name: scikit-learn
pipeline_tag: video-classification
language:
  - ko
tags:
  - korean-sign-language
  - sign-language-recognition
  - mediapipe
  - keypoint-classification
  - real-time
  - joblib
license: other
---

# Full-Dataset KSL Word Recognition Model

This repository contains a MediaPipe-keypoint Korean Sign Language word recognizer for the realtime-ksl-captioning backend.

## Current Artifact

- Hugging Face repo: `{repo_id}`
- Train split: AIHub `1.Training`
- Evaluation split: AIHub `2.Validation`
- Train samples: `{metrics.get("train_samples", "unknown")}`
- Validation samples: `{metrics.get("validation_samples", "unknown")}`
- Label count: `{metrics.get("label_count", len(labels))}`
- Best validation accuracy: `{best.get("accuracy", "unknown")}`
- Best macro F1: `{best.get("macro_f1", "unknown")}`
- Best candidate: sequence length `{candidate.get("sequence_length", "unknown")}`, classifier `{candidate.get("classifier", "unknown")}`, include deltas `{candidate.get("include_deltas", "unknown")}`

The full metric report is stored in `metrics_full_mediapipe.json`.

## Runtime Contract

```python
def load_model(model_dir: str, device: str):
    ...

def predict(model, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    ...
```

Each frame must be an RGB `numpy.ndarray` with shape `(height, width, 3)` and dtype `uint8`.

## Limitations

This model is trained from AIHub source videos using MediaPipe Holistic keypoints. It should be evaluated again on the actual demo webcam environment before presenting it as robust for real meetings.
""",
        encoding="utf-8",
    )
    return card_path


if __name__ == "__main__":
    main()
