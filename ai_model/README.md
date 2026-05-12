# KSL Word Recognition Model Repo

This folder is a Hugging Face-compatible model package for word-level Korean Sign Language recognition. It contains the model structure and inference contract; AIHub data preparation and training experiments can be done later without changing the backend contract.

## Runtime Structure

```text
RGB webcam frame
  -> YoloRoiExtractor
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
