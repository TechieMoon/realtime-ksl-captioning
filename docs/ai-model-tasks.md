# AI Model Tasks

이 문서는 AI 모델 담당자가 Codex로 바로 구현할 수 있도록 만든 작업 지시서입니다.

## Goal

Korean Sign Language webcam frames를 입력받아 단어 단위 한국어 caption을 반환하는 모델을 만들고, Hugging Face Hub에 업로드합니다. 백엔드 서버는 이 repo를 내려받아 RTX3090에서 직접 실행합니다.

## Hugging Face Repo Contract

모델 repo에는 최소한 다음 파일이 있어야 합니다.

```text
inference.py
labels.json
model weights/config/preprocessor files
README.md
```

`inference.py` must define:

```python
def load_model(model_dir: str, device: str):
    """Load weights and preprocessors from model_dir onto device."""


def predict(frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    """Return one prediction dict per input frame."""
```

The server also accepts this variant if you prefer to avoid module-level globals:

```python
def predict(model, frames: list, timestamps_ms: list[int | None]) -> list[dict]:
    ...
```

Each `frame` is an RGB `numpy.ndarray` with shape `(height, width, 3)` and dtype `uint8`.

Prediction output:

```python
[
    {
        "text": "안녕하세요",
        "words": [
            {
                "text": "안녕하세요",
                "confidence": 0.91,
                "start_ms": 1200,
                "end_ms": 1700,
            }
        ],
        "is_final": True,
    }
]
```

## MVP Model Expectations

- Start with a small fixed vocabulary that can run in real time.
- Target one signer, front-facing webcam, stable lighting, and 360p or 480p input.
- Return an empty text or low confidence when the sign is unknown.
- Prefer stable word output over rapid flickering predictions.
- Track latency per frame; first demo target is under 150 ms inference time on RTX3090 for one frame.

## Packaging Notes

- Use `safetensors` where possible for weights.
- Include preprocessing config so the server does not hard-code image size or normalization.
- If the model requires extra Python packages, list them in the model repo README and tell the backend owner before integration.
- If the Hugging Face repo is private, provide the repo id and confirm that the server has an `HF_TOKEN`.

## Acceptance Criteria

- `load_model(model_dir, "cuda:0")` loads on the RTX3090 server.
- `predict([rgb_frame], [timestamp_ms])` returns a list with one valid prediction.
- Repeated calls do not reload weights.
- The model can process at least 8 fps in the controlled MVP demo setup.

