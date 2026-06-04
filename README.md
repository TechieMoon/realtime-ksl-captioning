# realtime-ksl-captioning

Real-Time Word-Level Korean Sign Language Captioning System for Video Conferencing.

이 프로젝트는 농인 사용자의 노트북에서 webcam 영상을 서버로 보내고, 서버에서 Korean Sign Language 단어 자막을 생성한 뒤, 클라이언트가 자막을 webcam preview 위에 overlay해서 Zoom/Google Meet에 전달하는 MVP입니다.

## Architecture

```text
Desktop client webcam
  -> WebSocket JPEG frame sequence per word
  -> FastAPI backend on localhost or LAN host
  -> Mock or Hugging Face-loaded KSL word classifier
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

## Frontend Quickstart

Node.js 20.19 or newer is required by Vite. On Ubuntu, install Node 20 if the
system `node -v` is still Node 18:

```bash
sudo apt update
sudo apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

```text
http://localhost:5173
```

The frontend has a webcam dropdown and two server host modes:

- `localhost`: use this when frontend and backend run on the same computer.
- `IP address`: use this when the backend is on another computer in the same LAN.

Connect to the server first, then record one isolated word segment at a time. Press Space once to start recording a word and press Space again to stop recording and send that frame sequence to the backend.

## Local End-to-End Test

Run the backend and frontend in two separate terminals.

For a fresh Ubuntu checkout, run the setup script first:

```bash
cd realtime-ksl-captioning
./scripts/setup_ubuntu.sh
```

This script creates `server/.venv`, installs backend dependencies, copies
`server/.env.example` to `server/.env` if needed, checks Node.js 20.19+, and
installs frontend dependencies. The `.venv` directory itself is intentionally
not committed because virtual environments are machine-specific.

Terminal 1, backend:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If `python3 -m venv .venv` fails with `ensurepip is not available`, install the
Ubuntu venv package and recreate the virtual environment:

```bash
sudo apt update
sudo apt install python3.12-venv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

Terminal 2, frontend:

```bash
cd realtime-ksl-captioning/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173`, allow camera access, keep `Host Mode` set to
`localhost`, keep `Port` set to `8000`, then click `Connect`. The top bar should
show `Connected` and the server status should show `session_started`.

To test one word: click outside text inputs, press Space to start recording,
perform one sign word, then press Space again to stop and send the frame
sequence. The prediction appears as the video overlay caption. For a connection
smoke test without the real model, start the backend with `MODEL_BACKEND=mock`.

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

Default mode still uses the mock model for backend tests:

```powershell
$env:MODEL_BACKEND = "mock"
```

For the current Hugging Face word-classifier model:

```powershell
$env:MODEL_BACKEND = "huggingface"
$env:HF_MODEL_ID = "Seoyoung07/korean-sign-word-classifier-mediapipe"
$env:HF_MODEL_REVISION = "main"
$env:MODEL_DEVICE = "cpu"
$env:HF_TOKEN = "<optional-private-model-token>"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The adapter supports two Hugging Face repo formats:

- A repo with `inference.py`, `load_model(model_dir, device)`, and `predict(...)`.
- The current `Seoyoung07/korean-sign-word-classifier-mediapipe` repo with `best.pt` and `model/predict_word_classifier.py`.

The current model is an isolated-word classifier. It expects one word clip at a time, so the frontend records and sends one frame sequence per word.

## Local AI Model Package

The checked-in [ai_model](ai_model) folder is kept only for the legacy
MediaPipe MVP training/upload flow. The current end-to-end app loads the team
word-classifier model from Hugging Face:

```text
Seoyoung07/korean-sign-word-classifier-mediapipe
```

Unused CTC/gloss decoder code is not part of this GitHub repository. The old
untrained YOLO/VideoMAE scaffold was also removed so the repo only keeps model
paths that are actually exercised.

## Training

Training code lives in [training/README.md](training/README.md). AIHub dataset zip files and trained model artifacts are intentionally not tracked in GitHub.

The training script prints validation accuracy and a per-class report automatically, then writes `ai_model/mediapipe_mvp.joblib` for Hugging Face upload. It also writes `ai_model/metrics_mediapipe_mvp.json`, which should be uploaded to Hugging Face with the model artifact but not committed to GitHub.

For a class-presentation model that uses the full downloaded AIHub training and validation splits:

```powershell
$env:KSL_DATA_ROOT = "D:\수어 영상"
$env:KSL_CACHE_DIR = "D:\ksl_cache\full_mediapipe_features"
python training\train_full_mediapipe.py --time-budget-hours 12 --data-root "D:\수어 영상" --cache-dir "D:\ksl_cache\full_mediapipe_features"
python training\upload_mediapipe_to_hf.py --repo-id TechieMoon/realtime-ksl-captioning-mediapipe-mvp
```

The full trainer uses `1.Training` only for training and `2.Validation` only for evaluation. It scans every paired `video.zip` and `morpheme.zip`, tries multiple model candidates within the 12-hour budget, and keeps the best validation-accuracy artifact.

The full downloaded `real_word` split is large enough that the first strict all-data MediaPipe run may spend the whole 12 hours extracting features. Re-run the same command to resume from cache; do not delete `D:\ksl_cache\full_mediapipe_features`.

## WebSocket Protocol

Endpoint:

```text
ws://<server-host>:8000/ws/captions?session_id=<unique-session-id>
```

The client sends one JSON `start` message after connecting. For isolated-word inference, it then sends `segment_start`, the binary JPEG frame packets for that word, and `segment_end`.

Start message:

```json
{
  "type": "start",
  "width": 640,
  "height": 360,
  "fps": 8,
  "format": "jpeg",
  "client_name": "desktop-client"
}
```

Segment messages:

```json
{"type": "segment_start", "segment_id": "segment-1"}
```

```json
{"type": "segment_end", "segment_id": "segment-1"}
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
  "latency_ms": 18.4,
  "segment_id": "segment-1"
}
```

## Team Tasks

- Backend: `server/` FastAPI backend, WebSocket protocol, model adapter, GPU server runtime.
- Frontend client: webcam selection, isolated-word frame sequence capture, caption overlay, OBS Virtual Camera MVP. See [docs/frontend-client-tasks.md](docs/frontend-client-tasks.md).
- AI model: KSL word recognition model, Hugging Face repo packaging, `inference.py` contract. See [docs/ai-model-tasks.md](docs/ai-model-tasks.md).

## Tests

```powershell
cd server
pytest
```
