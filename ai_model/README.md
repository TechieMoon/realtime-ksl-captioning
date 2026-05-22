---
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

# KSL Word Recognition Model Repo

This folder is a Hugging Face-compatible model package for word-level Korean Sign Language recognition. It contains the model structure, inference contract, and the MediaPipe MVP runtime used by the backend.

## Current MVP Artifact

The repository loads a lightweight MediaPipe MVP artifact when `mediapipe_mvp.joblib` is present. This artifact is trained for a controlled one-person proof-of-concept, not for production-quality Korean Sign Language translation.

Current MVP vocabulary:

```text
수어, 좋다, 감사, 괜찮다, 싫다, 이해, 부탁, 모르다, 맞다, 힘
```

Measured locally on the development machine:

```text
Validation accuracy: 98.0% on a 50-sample held-out signer split
Training samples: 900 balanced samples, 10 labels, 5 camera angles
Mean inference latency: about 112.3 ms/frame on CPU
Approximate throughput: about 8.9 fps
```

Important limitation: this metric is from controlled AIHub word clips and a small 10-word vocabulary. It is suitable for a one-person demo, but it is not evidence of robust real-world meeting performance.

Training command used for the current artifact:

```powershell
python training\train_mediapipe_mvp.py --data-root "D:\수어 영상" --cache-dir "D:\ksl_cache\mediapipe_mvp_features" --max-per-label 90 --sequence-length 16 --angles F D U L R --classifier extra_trees --confidence-threshold 0.45
```

Benchmark command:

```powershell
python training\benchmark_mediapipe_mvp.py --data-root "D:\수어 영상" --cache-dir "D:\ksl_cache\mediapipe_mvp_features" --max-frames 120
```

The trained `mediapipe_mvp.joblib` file is intentionally not tracked in GitHub. It should be distributed through Hugging Face.

Hugging Face model repo used by the backend:

```text
TechieMoon/realtime-ksl-captioning-mediapipe-mvp
```

Backend environment:

```powershell
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "TechieMoon/realtime-ksl-captioning-mediapipe-mvp"
$env:MODEL_DEVICE = "cpu"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This MVP artifact does not contain the AIHub video dataset. It only contains the trained lightweight classifier and inference code.

## Runtime Structure

```text
RGB webcam frame
  -> MediaPipe MVP recognizer when mediapipe_mvp.joblib exists
     - resizes large RGB frames to max width 640 before MediaPipe
     - extracts pose and hand landmarks
     - classifies a short keypoint sequence with a lightweight classifier
  -> otherwise YoloRoiExtractor
     - uses yolo/yolo.pt when present and enabled
     - falls back to the full frame when no detector checkpoint exists
  -> VideoMAEWordClassifier
     - keeps a short frame clip buffer
     - loads a transformers VideoMAE checkpoint from videomae/ when present
     - returns unknown until enough clip frames and a checkpoint are available
  -> KslWordRecognizer
     - confidence thresholding
     - short-window smoothing
     - word-level caption JSON
```

The current repository intentionally does not include trained checkpoints. Add them later with this layout:

```text
ai_model/
  inference.py
  modeling.py
  labels.json
  model_config.json
  preprocessor_config.json
  yolo/
    yolo.pt
  videomae/
    config.json
    model.safetensors
    preprocessor_config.json
```

## Contract

```python
def load_model(model_dir: str, device: str):
    ...

def predict(model, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    ...
```

Each `frame` must be an RGB `numpy.ndarray` with shape `(height, width, 3)` and dtype `uint8`.

Output:

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

If no checkpoint exists, inference stays conservative and returns an empty caption:

```json
{
  "text": "",
  "words": [],
  "is_final": false
}
```

## Config

`model_config.json` controls the optional detector and classifier:

```json
{
  "roi_detector": {
    "enabled": false,
    "weights_path": "yolo/yolo.pt"
  },
  "classifier": {
    "checkpoint_path": "videomae",
    "clip_size": 8
  }
}
```

Set `roi_detector.enabled` to `true` after adding YOLO weights. The classifier automatically loads `videomae/` when that checkpoint directory exists.

## AIHub Training Fit

For AIHub 수어 영상 데이터, the later training side should create word-level clips from video annotations and fine-tune a VideoMAE classifier with label IDs aligned to `labels.json`. The resulting Hugging Face checkpoint can be saved into `videomae/`; no backend changes are required.

YOLO can be trained or fine-tuned separately for signer/hand/upper-body ROI detection, then exported as `yolo/yolo.pt`.

## Local Smoke Test

From the repository root:

```powershell
cd server
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "..\ai_model"
$env:MODEL_DEVICE = "cpu"
pytest tests/test_ai_model_contract.py tests/test_huggingface_adapter.py
```
