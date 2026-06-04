---
library_name: pytorch
pipeline_tag: video-classification
language:
  - ko
tags:
  - korean-sign-language
  - sign-language-recognition
  - mediapipe
  - keypoint-classification
  - real-time
license: other
---

# Local AI Model Utilities

This folder contains the checked-in model-side code that is useful for local
experiments and model maintenance.

The current end-to-end app still loads the team model from Hugging Face:

```text
Seoyoung07/korean-sign-word-classifier-mediapipe
```

That production path is configured in the backend. The files here are for local
reproduction, training utilities, and the legacy MediaPipe MVP upload flow.

## What Is Included

- `inference.py`: backend/Hugging Face contract entry point for the legacy local
  MediaPipe MVP artifact.
- `mediapipe_mvp.py`: legacy MediaPipe keypoint extraction and lightweight
  sequence classifier runtime.
- `keypoint_extractor.py`: MediaPipe Holistic mp4/frame extractor and 543-to-115
  keypoint selector.
- `preprocessing.py`: body-centered normalization, velocity, acceleration, and
  wrist-relative keypoint features.
- `word_classifier.py`: isolated-word keypoint classifier.
- `prepare_keypoints.py`: extracts `.npy` keypoint caches from mp4 clips.
- `train_word_classifier.py`: trains the isolated-word classifier with cross
  entropy.
- `predict_word_classifier.py`: runs top-k prediction for one mp4 clip or one
  pre-extracted `.npy` cache.
- `assets/holistic_landmarker.task`: MediaPipe Tasks model used when the
  installed MediaPipe package does not expose the legacy Holistic API.

Unused CTC/gloss decoder files from the teammate's `model/` folder were removed:
`ctc_decoder.py`, `gloss2text.py`, `keypoint_ctc.py`, and `pipeline.py` are not
part of the checked-in local package because the current classifier was not
trained with CTC.

## Runtime Contract

```python
def load_model(model_dir: str, device: str):
    ...

def predict(model, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    ...
```

Each frame must be an RGB `numpy.ndarray` with shape `(height, width, 3)` and
dtype `uint8`.

Prediction output:

```json
{
  "text": "안녕하세요",
  "words": [
    {
      "text": "안녕하세요",
      "confidence": 0.91,
      "start_ms": 1200,
      "end_ms": 1700
    }
  ],
  "is_final": true
}
```

If `mediapipe_mvp.joblib` is absent, `inference.py` returns empty predictions so
backend contract tests can run without a local checkpoint.

## Isolated Word Training Flow

Install local model dependencies:

```bash
python -m pip install -r ai_model/requirements.txt
```

Create keypoint caches:

```bash
python -m ai_model.prepare_keypoints   --split-root "1.Training"   --cache-root keypoint_cache   --manifest-out manifests/train.json   --skip-existing

python -m ai_model.prepare_keypoints   --split-root "2.Validation"   --cache-root keypoint_cache   --manifest-out manifests/val.json   --skip-existing
```

Train the isolated-word classifier:

```bash
python -m ai_model.train_word_classifier   --train-manifest manifests/train.json   --val-manifest manifests/val.json   --output-dir checkpoints/word_classifier   --epochs 20   --batch-size 16   --max-frames 64
```

Run a single video prediction:

```bash
python -m ai_model.predict_word_classifier sample.mp4   --checkpoint checkpoints/word_classifier/best.pt   --top-k 5
```

The default MediaPipe Tasks asset path is:

```text
ai_model/assets/holistic_landmarker.task
```
