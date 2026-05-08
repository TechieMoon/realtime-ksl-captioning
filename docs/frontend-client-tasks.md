# Frontend Client Tasks

이 문서는 클라이언트 담당자가 Codex로 바로 구현할 수 있도록 만든 작업 지시서입니다.

## Goal

농인 사용자의 노트북에서 webcam 영상을 읽고, 백엔드 WebSocket으로 JPEG frame을 보내며, 서버에서 받은 단어 자막을 영상 위에 overlay합니다. MVP에서는 native virtual camera driver를 직접 만들지 않고 OBS Virtual Camera를 사용합니다.

## Required UX

- 사용자가 서버 URL, session id, optional auth token을 입력할 수 있어야 합니다.
- webcam device를 선택하고 preview를 볼 수 있어야 합니다.
- caption overlay 위치, 글자 크기, 배경 불투명도를 조정할 수 있어야 합니다.
- 서버 연결 상태, frame drop 상태, 최근 caption latency를 화면에 보여야 합니다.
- OBS에서 창 캡처 또는 browser/source capture를 할 수 있도록 overlay된 최종 preview 창을 제공합니다.

## WebSocket Protocol

Endpoint:

```text
ws://<server-host>:8000/ws/captions?session_id=<unique-session-id>
```

Private demo 서버에서는 query string에 `token=<CAPTION_AUTH_TOKEN>`을 추가합니다.

Local network example:

1. Ask the backend owner for the server computer's local IPv4 address, for example `192.168.0.25`.
2. Confirm `http://192.168.0.25:8000/healthz` returns `{"status":"ok"}` from the client computer.
3. Use `ws://192.168.0.25:8000/ws/captions?session_id=demo-1` as the WebSocket URL.
4. If connection fails, ask the backend owner to check firewall access for TCP port `8000` and confirm the server is running with `--host 0.0.0.0`.

Connection flow:

1. Connect to WebSocket.
2. Receive a `status` event with `status=connected`.
3. Send one JSON start message.
4. Send each video frame as one binary WebSocket packet.
5. Render incoming `caption` events over the local preview.

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

Binary frame packet:

```text
4 bytes: unsigned big-endian metadata JSON byte length
N bytes: UTF-8 JSON metadata
M bytes: JPEG image bytes
```

Metadata JSON:

```json
{
  "frame_id": 123,
  "timestamp_ms": 15375,
  "width": 640,
  "height": 360,
  "format": "jpeg"
}
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

## Implementation Notes

- Start with 640x360 or 480x270 at 8-12 fps. Higher frame rates are unnecessary until the model proves it can keep up.
- JPEG quality around 70-85 is enough for the MVP.
- Keep sending frames even if captions arrive slowly; the server drops old frames to preserve real-time behavior.
- Reconnect automatically if the socket closes, but do not spam reconnects faster than once per second.
- Render the newest final caption until a newer caption arrives.
- For Zoom/Meet demo, open the overlay preview in a clean window and capture it in OBS, then start OBS Virtual Camera.

## Acceptance Criteria

- Client can connect to the mock backend and display rotating Korean sample words.
- Client sends valid frame packets that the backend smoke tests accept.
- Overlay text remains readable on bright and dark webcam backgrounds.
- OBS Virtual Camera can show the overlayed preview in Zoom or Google Meet.
