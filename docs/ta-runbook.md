# TA Runbook

This document is for a teaching assistant or an AI coding agent running this project on a fresh machine.

The expected demo flow is:

```text
webcam -> frontend word segment recording -> backend KSL word classifier
-> frontend rolling captions -> optional virtual camera -> Zoom
```

## What This Repo Contains

- `server/`: FastAPI backend, caption WebSocket, Hugging Face model adapter, virtual-camera WebSocket.
- `frontend/`: React/Vite browser client with webcam dropdown, server host settings, word capture controls, captions, and virtual-camera controls.
- `ai_model/`: local model/training utility code kept for reference and reproducibility.
- `scripts/setup_ubuntu.sh`: backend/frontend dependency setup for Ubuntu.
- `scripts/setup_virtual_camera_ubuntu.sh`: Ubuntu v4l2loopback setup for Zoom/Meet camera output.
- `reports/test100/`: saved evaluation reports for the 100 local isolated-word test videos.

The deployed inference model is downloaded from Hugging Face:

```text
Seoyoung07/korean-sign-word-classifier-mediapipe
```

The model is an isolated-word classifier. It does not continuously translate full sign-language sentences. The frontend sends one recorded word segment at a time.

## Requirements

Recommended OS:

- Ubuntu 22.04 or Ubuntu 24.04.

Required software:

- `git`
- Python 3.10 or newer
- Python venv support, usually `python3-venv`
- Node.js 20.19 or newer
- npm
- Chrome/Chromium browser with webcam permission
- Internet access for npm, pip, and Hugging Face download

Optional software:

- NVIDIA GPU and working PyTorch CUDA install for faster classifier inference.
- Zoom or Google Meet for virtual camera testing.
- `v4l2loopback-dkms` and `v4l2loopback-utils` for Linux virtual webcam output.

Hugging Face access:

- If the Hugging Face model repository is public, no token is needed.
- If it is private, create a read token and set `HF_TOKEN` in `server/.env`.
- Do not commit tokens to GitHub.

## Fresh Setup On Ubuntu

Clone the repo:

```bash
git clone https://github.com/TechieMoon/realtime-ksl-captioning.git
cd realtime-ksl-captioning
```

Install base packages:

```bash
sudo apt update
sudo apt install -y git curl python3-venv
```

Install Node.js 20:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v
```

`node -v` must be `v20.19.0` or newer. Node 18 will fail with Vite errors such as `CustomEvent is not defined`.

Run project setup:

```bash
./scripts/setup_ubuntu.sh
```

The script creates `server/.venv`, installs backend dependencies, copies `server/.env.example` to `server/.env` if missing, checks Node.js, and installs frontend dependencies.

If venv creation fails with `ensurepip is not available`, install the matching venv package and rerun:

```bash
sudo apt update
sudo apt install -y python3-venv
./scripts/setup_ubuntu.sh
```

## Backend Configuration

The backend reads settings from `server/.env` when it is started from the `server/` directory.

Default real-model settings:

```env
MODEL_BACKEND=huggingface
HF_MODEL_ID=Seoyoung07/korean-sign-word-classifier-mediapipe
HF_MODEL_REVISION=main
MODEL_DEVICE=auto
MODEL_TOP_K=5
SEQUENCE_TARGET_FPS=15
```

Use `MODEL_DEVICE=auto` for normal setup. It selects CUDA when PyTorch sees CUDA, otherwise CPU. Use `MODEL_DEVICE=cpu` for maximum compatibility. Use `MODEL_DEVICE=cuda` only when CUDA is definitely available; otherwise backend startup will fail.

If the Hugging Face repo is private:

```env
HF_TOKEN=hf_your_read_token_here
```

Mock mode for connectivity-only tests:

```env
MODEL_BACKEND=mock
```

Mock mode does not load MediaPipe or the Hugging Face model. It is useful when verifying frontend/backend WebSocket behavior.

## Start The Backend

Terminal 1:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check health from another terminal:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

Expected health response:

```json
{"status":"ok","service":"realtime-ksl-captioning"}
```

Expected ready response in real model mode:

```json
{
  "status": "ready",
  "model_backend": "huggingface",
  "model_id": "Seoyoung07/korean-sign-word-classifier-mediapipe",
  "model_revision": "main",
  "device": "auto",
  "adapter": "huggingface"
}
```

For a quick WebSocket smoke test with mock mode:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
MODEL_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then, in another terminal:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
python scripts/smoke_websocket_client.py
```

The smoke client should receive a caption event with the mock word `안녕하세요`.

## Start The Frontend

Terminal 2:

```bash
cd realtime-ksl-captioning/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

```text
http://localhost:5173
```

Allow browser camera permission.

Frontend settings for same-machine testing:

- `Webcam`: choose the physical webcam.
- `Host Mode`: `localhost`
- `Host / IP`: `127.0.0.1`
- `Port`: `8000`
- `Session ID`: any stable value, for example `demo-1`
- `Token`: leave blank unless `CAPTION_AUTH_TOKEN` is set in the backend.

Click `Connect`. The top bar should show `Connected`.

## Word Capture Controls

The model expects one isolated word at a time.

- Press `Space` once to start recording a word.
- Perform one sign word.
- Press `Space` again to stop recording and send the frame sequence.
- The predicted word is appended to the current caption line.
- Press `Enter` to move to the next sentence.
- The overlay keeps at most two sentences. When a third sentence starts, the first sentence is removed and the second sentence moves up.

Click outside text inputs before using `Space` or `Enter`.

## Zoom Virtual Camera Test

The browser cannot create an OS-level webcam device by itself. This project streams the frontend's overlay canvas back to the backend, and the backend writes it to a Linux v4l2loopback virtual webcam.

Prepare the virtual camera:

```bash
cd realtime-ksl-captioning
./scripts/setup_virtual_camera_ubuntu.sh
```

Default device:

```text
/dev/video20
```

Default Zoom camera name:

```text
KSL Caption Camera
```

If the script adds the user to the `video` group, log out and log back in, or run:

```bash
newgrp video
```

Verify the device:

```bash
ls -l /dev/video20
v4l2-ctl --list-devices
```

Start backend and frontend normally. In the frontend:

1. Connect to the caption server.
2. Keep virtual-camera device as `/dev/video20`.
3. Click `Start` in the `Virtual Camera` section.
4. Open or restart Zoom.
5. Choose `KSL Caption Camera` as the camera.

If Zoom was already open before creating `/dev/video20`, restart Zoom.

## Local Network Setup

If backend and frontend run on different computers:

1. Start backend on the server computer with `--host 0.0.0.0`.
2. Find the server computer's LAN IP.
3. Run the frontend on the client computer, preferably at `http://localhost:5173`.
4. In the frontend, set `Host Mode` to `IP address` and enter the backend IP.

Do not use the backend computer's `localhost` address from another computer. `localhost` always means the current machine.

Camera access note:

- Browser webcam access is reliable on `http://localhost`.
- If serving the frontend over a raw LAN IP using plain HTTP, some browsers may block webcam access because it is not a secure context.

## Validation Commands

Backend tests:

```bash
cd realtime-ksl-captioning/server
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Python compile check:

```bash
cd realtime-ksl-captioning
server/.venv/bin/python -m compileall server/app server/tests server/scripts scripts
```

Frontend checks:

```bash
cd realtime-ksl-captioning/frontend
npm run lint
npm run build
```

Test-100 report already committed:

```text
reports/test100/test100_report_20260606_171322_ko.md
```

Recorded result for the current updated Hugging Face weight:

- Top-1 correct: 60 / 100
- Top-5 contains expected: 71 / 100
- Inference errors: 0

## Troubleshooting

### `source .venv/bin/activate: No such file or directory`

The virtual environment has not been created yet.

```bash
cd realtime-ksl-captioning
./scripts/setup_ubuntu.sh
```

### `ensurepip is not available`

Install Python venv support:

```bash
sudo apt update
sudo apt install -y python3-venv
```

Then recreate the venv:

```bash
cd realtime-ksl-captioning/server
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Vite fails on Node 18

Install Node.js 20.19 or newer:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v
```

### Backend fails downloading Hugging Face model

Check internet access and Hugging Face permissions. If the model is private, set a read token:

```bash
cd realtime-ksl-captioning/server
nano .env
```

Add:

```env
HF_TOKEN=hf_your_read_token_here
```

Then restart backend.

### CUDA requested but unavailable

If startup fails with `MODEL_DEVICE=cuda was requested, but CUDA is not available`, edit `server/.env`:

```env
MODEL_DEVICE=auto
```

or:

```env
MODEL_DEVICE=cpu
```

Then restart backend.

### Frontend shows disconnected

Check:

- Backend terminal is still running.
- Backend started on port `8000`.
- Frontend `Host Mode`, `Host / IP`, and `Port` match the backend.
- `curl http://localhost:8000/healthz` works on the same computer.

### Webcam dropdown is empty

Check:

- Browser camera permission is allowed.
- Webcam is not exclusively used by another application.
- The page is opened at `http://localhost:5173`.
- Unplug and reconnect the webcam, then refresh.

### Virtual Camera Start immediately disconnects

Check:

```bash
ls -l /dev/video20
groups
v4l2-ctl --list-devices
```

If the user is not in the `video` group:

```bash
sudo usermod -aG video "$USER"
```

Then log out and log back in, or run `newgrp video`.

Also confirm the backend is running, because the frontend sends virtual-camera frames to the backend WebSocket.

### Zoom does not show `KSL Caption Camera`

Check:

- `/dev/video20` exists.
- `v4l2-ctl --list-devices` shows `KSL Caption Camera`.
- Zoom was restarted after creating the virtual camera.
- No other app is already consuming the virtual camera in a conflicting way.

## Notes For AI Agents

When reproducing this project automatically:

1. Start from `README.md` and this runbook.
2. Do not commit `.venv`, `node_modules`, Hugging Face tokens, downloaded datasets, or local cache directories.
3. Use mock backend first to verify WebSocket and frontend behavior.
4. Switch to `MODEL_BACKEND=huggingface` only after the mock path works.
5. For Zoom testing, create `/dev/video20` before opening Zoom.
6. Use `git status -sb` before committing, and stage only intentional documentation/code changes.
