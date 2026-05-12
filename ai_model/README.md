# KSL Word Recognition Model Repo

This folder is packaged so it can be uploaded directly to Hugging Face Hub and loaded by the FastAPI backend.

Required files:

```text
inference.py
labels.json
preprocessor_config.json
model_config.json
README.md
```

The current implementation is an inference pipeline scaffold. It validates RGB `numpy.ndarray` webcam frames, extracts light real-time motion features, smooths predictions across frames, and returns conservative unknown captions until trained weights or prototype features are added.

## Contract

```python
def load_model(model_dir: str, device: str):
    ...

def predict(model, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    ...
```

Each returned prediction has:

```json
{
  "text": "",
  "words": [],
  "is_final": false
}
```

When a trained class is emitted, `words` contains one word-level caption with confidence and timestamp bounds.

## Local Smoke Test

From the repository root:

```powershell
cd server
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "..\ai_model"
$env:MODEL_DEVICE = "cpu"
pytest tests/test_ai_model_contract.py tests/test_huggingface_adapter.py
```

## Future YOLO-Assisted VideoMAE Path

The planned production path can fit behind the same `inference.py` functions:

1. Run a hand/person detector such as YOLO to crop or mask the signing region.
2. Build a short temporal clip from recent frames.
3. Run a VideoMAE-style word classifier on the clip.
4. Smooth logits over a short window to prevent flickering captions.
5. Return low confidence or empty text for unknown signs.

Keep trained weights in `safetensors` when possible, and add any new runtime packages here plus a note in this README before the backend is deployed.
