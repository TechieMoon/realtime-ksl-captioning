# realtime-ksl-captioning

Real-Time Word-Level Korean Sign Language Captioning System for Video Conferencing.

이 프로젝트는 농인 사용자의 노트북에서 webcam 영상을 서버로 보내고, 서버에서 Korean Sign Language 단어 자막을 생성한 뒤, 클라이언트가 자막을 webcam preview 위에 overlay해서 Zoom/Google Meet에 전달하는 MVP입니다.

## Architecture

```text
Desktop client webcam
  -> WebSocket JPEG frames
  -> FastAPI backend on RTX3090 server
  -> Mock or Hugging Face-loaded KSL model
  -> WebSocket caption events
  -> Desktop client overlay preview
  -> OBS Virtual Camera
  -> Zoom / Google Meet
```

Hugging Face는 실시간 inference 서버가 아닙니다. AI 팀원이 모델을 Hugging Face Hub에 업로드하면, 백엔드 서버가 해당 repo를 다운로드하고 RTX3090에서 직접 실행합니다.

## Backend Quickstart

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health checks:

```text
GET http://localhost:8000/healthz
GET http://localhost:8000/readyz
```

Smoke test:

```powershell
cd server
python scripts/smoke_websocket_client.py
```

## Local Network Test

서버 컴퓨터와 클라이언트 컴퓨터를 따로 두고 테스트할 때는 두 컴퓨터가 같은 Wi-Fi 또는 같은 LAN에 있어야 합니다.

On the RTX3090 server computer:

```powershell
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find the server computer's local IPv4 address:

```powershell
ipconfig
```

Example:

```text
IPv4 Address . . . . . . . . . . . : 192.168.0.25
```

From the client computer, open this health check first:

```text
http://192.168.0.25:8000/healthz
```

If it returns `{"status":"ok"}`, connect the client WebSocket to:

```text
ws://192.168.0.25:8000/ws/captions?session_id=demo-1
```

If the health check fails:

- Confirm both computers are on the same local network.
- Allow inbound TCP traffic on port `8000` in the server computer's firewall.
- Check that the server was started with `--host 0.0.0.0`, not `--host 127.0.0.1`.
- Do not use the server's `localhost` address from the client computer; `localhost` always points to the current machine.

## Model Configuration

Default mode uses the mock model:

```powershell
$env:MODEL_BACKEND = "mock"
```

When the AI model repo is ready:

```powershell
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "org-or-user/model-name"
$env:HF_MODEL_REVISION = "main"
$env:MODEL_DEVICE = "cuda:0"
$env:HF_TOKEN = "<optional-private-model-token>"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The Hugging Face model repo must include `inference.py` with `load_model(model_dir, device)` and `predict(...)`. See [docs/ai-model-tasks.md](docs/ai-model-tasks.md).

## Checked-In AI Model Package

This repository now includes a local Hugging Face-compatible model package in [ai_model](ai_model). It is intentionally inference-first: training can happen later, then the trained weights/config can be added behind the same `inference.py` contract.

Local backend test with the checked-in model package:

```powershell
cd server
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "..\ai_model"
$env:MODEL_DEVICE = "cpu"
pytest tests/test_ai_model_contract.py tests/test_huggingface_adapter.py
```

Planned production direction:

```text
Webcam video -> YOLO/ROI detection -> VideoMAE-style temporal classifier -> KSL word captions
Microphone audio -> Whisper/faster-whisper ASR -> speech captions
Caption manager -> merge by timestamps -> overlay / virtual camera output
```

The current AI package covers the KSL model contract and conservative real-time inference scaffold. Audio ASR and caption merging can be added as separate backend/client tasks without changing this model contract.

## WebSocket Protocol

Endpoint:

```text
ws://<server-host>:8000/ws/captions?session_id=<unique-session-id>
```

The client sends one JSON `start` message, then binary JPEG frame packets.

Start message:

```json
{
  "type": "start",
  "width": 640,
  "height": 360,
  "fps": 10,
  "format": "jpeg",
  "client_name": "desktop-client"
}
```

Binary packet format:

```text
4 bytes: unsigned big-endian metadata JSON byte length
N bytes: UTF-8 JSON metadata
M bytes: JPEG image bytes
```

Caption event:

```json
{
  "type": "caption",
  "session_id": "demo-1",
  "frame_id": 123,
  "text": "안녕하세요",
  "words": [
    {
      "text": "안녕하세요",
      "confidence": 0.91,
      "start_ms": 15375,
      "end_ms": 15875
    }
  ],
  "is_final": true,
  "latency_ms": 18.4
}
```

## Team Tasks

- Backend: `server/` FastAPI backend, WebSocket protocol, model adapter, GPU server runtime.
- Frontend client: webcam capture, JPEG frame streaming, caption overlay, OBS Virtual Camera MVP. See [docs/frontend-client-tasks.md](docs/frontend-client-tasks.md).
- AI model: KSL word recognition model, Hugging Face repo packaging, `inference.py` contract. See [docs/ai-model-tasks.md](docs/ai-model-tasks.md).

## Tests

```powershell
cd server
pytest
```
