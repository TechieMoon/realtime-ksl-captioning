# realtime-ksl-captioning

Real-Time Word-Level Korean Sign Language Captioning System for Video Conferencing.

이 프로젝트는 농인 사용자의 노트북에서 webcam 영상을 서버로 보내고, 서버에서 Korean Sign Language 단어 자막을 생성한 뒤, 클라이언트가 자막을 webcam preview 위에 overlay해서 Zoom/Google Meet에 전달하는 MVP입니다.

## Submission Contents For TA Review

- [Execution Environment](#execution-environment)
- [How To Run](#how-to-run)
- [Reproducibility Scope](#reproducibility-scope)
- [AI Tools Used](#ai-tools-used)
- [Baseline Sources And Existing Components](#baseline-sources-and-existing-components)
- [TA / AI Agent Quickstart](#ta--ai-agent-quickstart)

## Execution Environment

Tested local environment:

- OS: Ubuntu 24.04-class Linux environment
- Python: 3.12.3
- Node.js: 20.20.2
- npm: 10.8.2
- GPU used for reported evaluation runs: NVIDIA RTX 3090 with CUDA-enabled PyTorch
- CPU-only inference is supported by setting `MODEL_DEVICE=cpu` or `MODEL_DEVICE=auto`, but it is slower.

Backend requirements are in [server/requirements.txt](server/requirements.txt). Key tested backend library versions:

| Package | Tested version | Requirement range |
|---|---:|---|
| `fastapi` | 0.136.3 | `>=0.115,<1.0` |
| `huggingface-hub` | 0.36.2 | `>=0.25,<1.0` |
| `mediapipe` | 0.10.35 | `>=0.10.35,<1.0` |
| `numpy` | 2.4.6 | `>=2.0,<3.0` |
| `opencv-python-headless` | 4.13.0.92 | `>=4.10,<5.0` |
| `pillow` | 11.3.0 | `>=10.4,<12.0` |
| `pydantic-settings` | 2.14.1 | `>=2.4,<3.0` |
| `pyvirtualcam` | 0.15.0 | `>=0.12,<1.0` |
| `torch` | 2.12.0 | `>=2.3,<3.0` |
| `uvicorn` | 0.48.0 | `>=0.30,<1.0` |

Frontend requirements are in [frontend/package.json](frontend/package.json). Important frontend dependencies:

- React 19
- Vite 8
- TypeScript 6
- Node.js `>=20.19.0`

The setup helper is:

```bash
./scripts/setup_ubuntu.sh
```

It creates `server/.venv`, installs backend dependencies, copies `server/.env.example` to `server/.env` if needed, verifies Node.js 20.19+, and installs frontend dependencies.

## How To Run

### 1. Install And Set Up

```bash
git clone https://github.com/TechieMoon/realtime-ksl-captioning.git
cd realtime-ksl-captioning

sudo apt update
sudo apt install -y git curl python3-venv

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

./scripts/setup_ubuntu.sh
```

### 2. Run Backend

Real model mode:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected health checks:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

Expected output shape:

```json
{"status":"ok","service":"realtime-ksl-captioning"}
```

`/readyz` should report `model_backend` as `huggingface` and `status` as `ready`.

Mock connectivity mode:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
MODEL_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
python scripts/smoke_websocket_client.py
```

Expected mock smoke output includes a `caption` event with text `안녕하세요`.

### 3. Run Frontend

```bash
cd realtime-ksl-captioning/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

```text
http://localhost:5173
```

Use the webcam dropdown, keep server mode as `localhost`, set port `8000`, and click `Connect`.

Word capture controls:

- `Space`: start one isolated-word recording.
- `Space`: stop recording and send that word segment to the backend.
- `Enter`: move to the next caption sentence.
- The overlay keeps at most two caption sentences.

### 4. Reproduce Evaluation Results

The trained model weights are public on Hugging Face:

```text
https://huggingface.co/Seoyoung07/korean-sign-word-classifier-mediapipe
```

The evaluation datasets are public on Hugging Face:

```text
https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-test-100
https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60
```

Download the evaluation datasets:

```bash
cd realtime-ksl-captioning

server/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Seoyoung07/korean-sign-word-classifier-mediapipe-test-100",
    repo_type="dataset",
    local_dir="data/hf-test-100",
)

snapshot_download(
    repo_id="Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60",
    repo_type="dataset",
    local_dir="data/hf-self-made-60",
)
PY
```

Run Test-100 evaluation:

```bash
cd realtime-ksl-captioning
server/.venv/bin/python scripts/evaluate_video_folder.py \
  data/hf-test-100/videos \
  --output-dir reports/reproduced-test100 \
  --device auto \
  --top-k 5
```

Expected aggregate output:

```text
top1=60/100 top5=71/100 errors=0
```

Run Self-made-60 evaluation:

```bash
cd realtime-ksl-captioning
server/.venv/bin/python scripts/evaluate_video_folder.py \
  data/hf-self-made-60/videos \
  --output-dir reports/reproduced-self-made-60 \
  --device auto \
  --top-k 5
```

Expected aggregate output:

```text
top1=5/60 top5=7/60 errors=0
```

The committed reference reports are:

- [reports/test100/README.md](reports/test100/README.md)
- [reports/self-made-60/README.md](reports/self-made-60/README.md)
- [docs/huggingface-datasets.md](docs/huggingface-datasets.md)

### 5. Run Zoom Virtual Camera Demo

```bash
cd realtime-ksl-captioning
./scripts/setup_virtual_camera_ubuntu.sh
```

Then run backend and frontend, click `Virtual Camera` `Start` in the frontend, and select `KSL Caption Camera` in Zoom.

## Reproducibility Scope

This submission supports end-to-end inference and evaluation reproduction, not full training-from-scratch reproduction.

What can be reproduced after downloading this repo:

- Backend startup with mock model and real Hugging Face model.
- Frontend webcam word-capture workflow.
- Localhost end-to-end inference from webcam to caption.
- Optional Zoom virtual camera output on Ubuntu with `v4l2loopback`.
- Test-100 evaluation result: Top-1 `60 / 100`, Top-5 `71 / 100`, errors `0`.
- Self-made-60 evaluation result: Top-1 `5 / 60`, Top-5 `7 / 60`, errors `0`.

What is intentionally not fully reproduced from scratch:

- Full training of the epoch-41 checkpoint.

Reason:

- The model was trained on Korean sign-language video data from AIHub. The raw AIHub training data is not redistributed in this GitHub repo because it is large and subject to AIHub access/license terms.
- The full local keypoint cache used during continued training was also large and machine-local, so it is not committed to GitHub.
- Full training would require separate AIHub dataset access, significant storage, MediaPipe keypoint extraction time, and GPU training time.

Accepted reproducibility artifact for this submission:

- Trained model weights on Hugging Face.
- Public evaluation datasets on Hugging Face.
- Inference and evaluation code in this GitHub repo.
- Committed evaluation reports and machine-readable CSV/JSON files.

Therefore, the submitted reproducible result is the inference/evaluation result for the uploaded weights on the two public evaluation datasets, not the complete training process from raw AIHub videos.

## AI Tools Used

- AI tool used: Codex.

Where Codex was used:

- Backend/frontend implementation support.
- WebSocket protocol and virtual-camera implementation support.
- Local debugging and test execution support.
- Evaluation script updates and result report generation.
- Hugging Face model/dataset upload packaging support.
- README and TA runbook documentation support.

Codex was not used to create the training videos or to generate model labels. The evaluation videos are real MP4 files, and model predictions are produced by the submitted Hugging Face checkpoint and the repository inference code.

## Baseline Sources And Existing Components

This project does not include a separate comparison baseline experiment. The submitted result is the current MediaPipe-keypoint word-classifier inference/evaluation pipeline.

Existing components used:

- MediaPipe Holistic Landmarker is used for the first-stage keypoint extraction from video frames. It combines pose, face, and hand landmarking for full-body landmark extraction. See the official Google AI Edge MediaPipe Holistic Landmarker documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
- MediaPipe framework source/documentation: https://github.com/google-ai-edge/mediapipe
- Hugging Face Hub is used for trained model and evaluation dataset distribution. Dataset upload/documentation follows Hugging Face dataset repository conventions: https://huggingface.co/docs/hub/en/datasets-adding
- Current model repo: https://huggingface.co/Seoyoung07/korean-sign-word-classifier-mediapipe

The neural classifier is trained on extracted keypoint sequences, not raw RGB frames. The current inference wrapper performs:

```text
MP4 video
-> MediaPipe Holistic Landmarker keypoint extraction
-> [T, 115, 4] keypoint sequence
-> PyTorch word classifier
-> Korean word prediction
```

## TA / AI Agent Quickstart

조교가 새 Ubuntu 컴퓨터에서 바로 실행할 때는 이 순서대로 진행하면 됩니다.
더 자세한 요구사항, 검증 명령, 문제 해결은 [docs/ta-runbook.md](docs/ta-runbook.md)를 참고하세요.

### 1. Requirements

- Ubuntu 22.04/24.04 권장
- Python 3.10 이상, `python3-venv`, `pip`
- Node.js 20.19 이상
- Webcam이 연결된 Chrome/Chromium 계열 브라우저
- 인터넷 연결: Python/npm 패키지 설치 및 Hugging Face 모델 다운로드용
- 실제 모델 실행 시 Hugging Face repo 접근 권한
  - public repo이면 token 없이 실행 가능
  - private repo이면 read token을 `server/.env`의 `HF_TOKEN`에 넣어야 함
- Zoom virtual camera 테스트 시 Linux `v4l2loopback` 지원 필요
- CUDA GPU는 선택사항입니다. 없으면 `MODEL_DEVICE=cpu` 또는 `auto`로 CPU 실행이 가능합니다.

### 2. Fresh Clone Setup

```bash
git clone https://github.com/TechieMoon/realtime-ksl-captioning.git
cd realtime-ksl-captioning

# Ubuntu에서 venv 패키지가 없다면 먼저 설치
sudo apt update
sudo apt install -y python3-venv curl git

# Node.js 20.19+ 필요. node -v가 20.19 미만이면 설치
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

./scripts/setup_ubuntu.sh
```

`scripts/setup_ubuntu.sh`는 다음 작업을 수행합니다.

- `server/.venv` 생성
- backend dependency 설치
- `server/.env.example`을 `server/.env`로 복사
- Node.js version 확인
- frontend dependency 설치

### 3. Start Backend

Terminal 1:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate

# 실모델 기본값: MODEL_BACKEND=huggingface
# GPU가 없으면 server/.env에서 MODEL_DEVICE=cpu 또는 auto로 둡니다.
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend가 켜진 뒤 다른 터미널에서 확인:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

실모델 다운로드나 Hugging Face 권한 문제를 배제하고 WebSocket만 먼저 확인하려면:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
MODEL_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

Terminal 2:

```bash
cd realtime-ksl-captioning/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

```text
http://localhost:5173
```

Frontend 설정:

- Webcam dropdown에서 사용할 카메라 선택
- Server Host Mode: `localhost`
- Host/IP: `127.0.0.1`
- Port: `8000`
- `Connect` 클릭

사용법:

- `Space`: 단어 녹화 시작
- 수어 단어 수행
- `Space`: 단어 녹화 종료 및 backend 전송
- 예측 단어가 자막에 append됨
- `Enter`: 다음 문장 시작
- 자막은 최대 두 문장 유지

### 5. Zoom Virtual Camera

Zoom에서 자막 overlay 영상을 카메라로 선택하려면 Ubuntu에서 가상 카메라를 준비합니다.

```bash
cd realtime-ksl-captioning
./scripts/setup_virtual_camera_ubuntu.sh
```

로그아웃/로그인 또는 `newgrp video`가 필요할 수 있습니다. 그 다음:

1. Backend와 frontend를 실행합니다.
2. Frontend에서 caption server에 `Connect`합니다.
3. `Virtual Camera` device를 `/dev/video20`으로 둡니다.
4. `Virtual Camera` 섹션의 `Start`를 누릅니다.
5. Zoom에서 `KSL Caption Camera`를 선택합니다.

### 6. Verification Commands

```bash
# Backend unit tests
cd realtime-ksl-captioning/server
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest

# Backend compile check
cd realtime-ksl-captioning
server/.venv/bin/python -m compileall server/app server/tests server/scripts scripts

# Frontend checks
cd realtime-ksl-captioning/frontend
npm run lint
npm run build
```

Latest local Test-100 report:

- [reports/test100/test100_report_20260606_171322_ko.md](reports/test100/test100_report_20260606_171322_ko.md)
- Top-1: 60 / 100
- Top-5: 71 / 100
- Inference errors: 0

Latest self-made-60 report:

- [reports/self-made-60/README.md](reports/self-made-60/README.md)
- Top-1: 5 / 60
- Top-5: 7 / 60
- Inference errors: 0

Public Hugging Face evaluation datasets:

- [Test-100 dataset](https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-test-100)
- [Self-made-60 dataset](https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60)
- Upload and verification notes: [docs/huggingface-datasets.md](docs/huggingface-datasets.md)
- AI tool used: Codex

## Architecture

```text
Desktop client webcam
  -> WebSocket JPEG frame sequence per word
  -> FastAPI backend on localhost or LAN host
  -> Mock or Hugging Face-loaded KSL word classifier
  -> WebSocket caption events
  -> Desktop client overlay preview and overlay canvas
  -> Backend virtual-camera WebSocket
  -> v4l2loopback virtual webcam device
  -> Zoom / Google Meet
```

Hugging Face는 실시간 inference 서버가 아닙니다. AI 팀원이 모델을 Hugging Face Hub에 업로드하면, 백엔드 서버가 해당 repo를 다운로드하고 로컬 CPU/GPU에서 직접 실행합니다.

## Backend Quickstart

```bash
cd realtime-ksl-captioning/server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp -n .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health checks:

```text
GET http://localhost:8000/healthz
GET http://localhost:8000/readyz
```

Smoke test:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
MODEL_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
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

Connect to the server first, then record one isolated word segment at a time.
Press Space once to start recording a word and press Space again to stop
recording and send that frame sequence to the backend. Predicted words are
appended to the current caption sentence with spaces. Press Enter to start the
next sentence. The overlay keeps at most two sentences; when a third sentence is
started, the older first sentence is removed and the second sentence moves up.

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
sequence. The prediction is appended to the video overlay caption. Press Enter
to move to the next sentence. For a connection smoke test without the real
model, start the backend with `MODEL_BACKEND=mock`.

## Zoom Virtual Camera Test

The browser cannot create a selectable OS webcam device by itself. For Zoom or
Google Meet, the frontend draws the webcam and two-line caption overlay into a
hidden canvas, sends those JPEG frames to the backend, and the backend writes
them to a local v4l2loopback virtual webcam.

On Ubuntu, prepare a virtual webcam device:

```bash
cd realtime-ksl-captioning
cd server
source .venv/bin/activate
pip install -r requirements.txt
cd ..
./scripts/setup_virtual_camera_ubuntu.sh
```

The default device is:

```text
/dev/video20
```

Start backend and frontend as in the local end-to-end test. In the frontend:

1. Connect the caption server.
2. Keep `Virtual Camera` device as `/dev/video20`.
3. Click `Start` in the `Virtual Camera` section.
4. Open Zoom and select `KSL Caption Camera` as the camera.

If Zoom does not show the virtual camera, run:

```bash
v4l2-ctl --list-devices
```

If `/dev/video20` exists but the backend cannot open it, confirm your user is in
the `video` group and log out/in after group changes:

```bash
groups
```

If `./scripts/setup_virtual_camera_ubuntu.sh` is slow during `apt update`, check
for stale third-party APT sources. For example, uninstalling NordVPN can leave
`/etc/apt/sources.list.d/nordvpn.list`, so APT still waits on
`repo.nordvpn.com`. If NordVPN is not needed, disable that source:

```bash
sudo mv /etc/apt/sources.list.d/nordvpn.list /etc/apt/sources.list.d/nordvpn.list.disabled
sudo apt-get update
```

## Local Network Test

서버 컴퓨터와 클라이언트 컴퓨터를 따로 두고 테스트할 때는 두 컴퓨터가 같은 Wi-Fi 또는 같은 LAN에 있어야 합니다.

On the backend server computer:

```bash
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find the server computer's local IPv4 address:

```bash
hostname -I
# or
ip -4 addr
```

Example:

```text
192.168.0.25
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

```bash
MODEL_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For the current Hugging Face word-classifier model:

```bash
export MODEL_BACKEND=huggingface
export HF_MODEL_ID=Seoyoung07/korean-sign-word-classifier-mediapipe
export HF_MODEL_REVISION=main
export MODEL_DEVICE=auto
# export HF_TOKEN=<optional-private-model-read-token>
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The adapter supports two Hugging Face repo formats:

- A repo with `inference.py`, `load_model(model_dir, device)`, and `predict(...)`.
- The current `Seoyoung07/korean-sign-word-classifier-mediapipe` repo with `best.pt` and `model/predict_word_classifier.py`.

The current model is an isolated-word classifier. It expects one word clip at a time, so the frontend records and sends one frame sequence per word.

## Local AI Model Package

The checked-in [ai_model](ai_model) folder now contains the legacy MediaPipe MVP
runtime plus the teammate's isolated-word classifier training utilities. The
current end-to-end app loads the team word-classifier model from Hugging Face:

```text
Seoyoung07/korean-sign-word-classifier-mediapipe
```

The top-level `model/` folder was intentionally merged into `ai_model/` and
removed. Unused CTC/gloss decoder files were also removed because the current
word classifier was not trained with CTC. For local word-classifier training or
single-video prediction, see [ai_model/README.md](ai_model/README.md).

## Training

This section is kept for legacy and optional training context. It is not the
reproducibility path for the submitted epoch-41 Hugging Face checkpoint. For
grading, use the inference/evaluation commands in [How To Run](#how-to-run) and
the scope described in [Reproducibility Scope](#reproducibility-scope).

Legacy MediaPipe MVP training code lives in [training/README.md](training/README.md). The isolated-word keypoint classifier utilities live in [ai_model/README.md](ai_model/README.md). AIHub dataset zip files, local keypoint caches, and trained model artifacts are intentionally not tracked in GitHub.

The legacy training script prints validation accuracy and a per-class report automatically, then writes `ai_model/mediapipe_mvp.joblib` for Hugging Face upload. It also writes `ai_model/metrics_mediapipe_mvp.json`, which should be uploaded to Hugging Face with the model artifact but not committed to GitHub.

For historical local experiments with the full downloaded AIHub training and
validation splits:

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

Virtual camera endpoint:

```text
ws://<server-host>:8000/ws/virtual-camera?session_id=<unique-session-id>&device=/dev/video20
```

It uses the same `start` message and binary JPEG packet format. The JPEG frames
should already contain the final video image with captions drawn into it.

## Team Tasks

- Backend: `server/` FastAPI backend, WebSocket protocol, model adapter, GPU server runtime.
- Frontend client: webcam selection, isolated-word frame sequence capture, rolling two-sentence caption overlay, virtual-camera frame streaming. See [docs/frontend-client-tasks.md](docs/frontend-client-tasks.md).
- AI model: KSL word recognition model, Hugging Face repo packaging, `inference.py` contract. See [docs/ai-model-tasks.md](docs/ai-model-tasks.md).

## Tests

```powershell
cd server
pytest
```
