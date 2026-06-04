# KSL Caption Frontend

React/Vite client for local Korean Sign Language caption tests.

## Run

```bash
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

```text
http://localhost:5173
```

## Controls

- Space starts or stops one word recording and sends that frame sequence to the backend.
- Caption words are appended to the current sentence with spaces.
- Enter starts the next sentence.
- The overlay keeps two sentences. Starting another sentence drops the oldest one.

## Virtual Camera

The frontend sends overlay-rendered canvas frames to:

```text
ws://<server-host>:8000/ws/virtual-camera?session_id=<session>-virtual-camera&device=/dev/video20
```

Prepare `/dev/video20` with `scripts/setup_virtual_camera_ubuntu.sh`, then
select `KSL Caption Camera` in Zoom or Google Meet.
