# Frontend Client Tasks

이 문서는 클라이언트 담당자가 Codex로 바로 구현할 수 있도록 만든 작업 지시서입니다.

## Goal

농인 사용자의 노트북에서 webcam 영상을 읽고, 백엔드 WebSocket으로 JPEG frame sequence를 보내며, 서버에서 받은 단어 자막을 영상 위에 overlay합니다. Zoom/Meet 송출용으로는 overlay canvas frame을 백엔드의 virtual-camera WebSocket으로 보내고, 백엔드가 v4l2loopback 장치에 씁니다.

## Required UX

- 사용자가 서버 URL, session id, optional auth token을 입력할 수 있어야 합니다.
- webcam device를 선택하고 preview를 볼 수 있어야 합니다.
- caption overlay 글자 크기를 조정할 수 있어야 합니다.
- Space로 단어 recording을 시작/종료하고, Enter로 새 문장을 시작할 수 있어야 합니다.
- 자막은 단어 단위로 현재 문장에 띄어쓰기로 누적되고, 최대 두 문장만 유지해야 합니다.
- 서버 연결 상태, frame drop 상태, 최근 caption latency를 화면에 보여야 합니다.
- Zoom/Meet에서 선택할 수 있도록 백엔드 virtual-camera endpoint로 overlay frame을 송출할 수 있어야 합니다.

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
4. Send one word segment with `segment_start`, binary frame packets, and `segment_end`.
5. Append incoming `caption` text to the current sentence.
6. Render the rolling two-sentence caption over the local preview.

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

Virtual camera endpoint:

```text
ws://<server-host>:8000/ws/virtual-camera?session_id=<unique-session-id>&device=/dev/video20
```

This endpoint uses the same `start` message and binary JPEG packet format. The
frames sent here are not raw webcam frames; they are final overlay canvas frames.

## Implementation Notes

- Start with 640x360 or 480x270 at 8 fps. Higher frame rates are unnecessary until the model proves it can keep up.
- JPEG quality around 70-85 is enough for the MVP.
- Keep sending virtual-camera frames at the selected FPS while that socket is connected.
- Reconnect automatically if the socket closes, but do not spam reconnects faster than once per second.
- Do not clear captions when a word segment is sent. Only append recognized words or move sentence lines when Enter is pressed.
- For Zoom/Meet demo, start v4l2loopback, connect the frontend virtual camera panel to `/dev/video20`, then select `KSL Caption Camera` in the meeting app.

## Acceptance Criteria

- Client can connect to the mock backend and display rotating Korean sample words.
- Client sends valid frame packets that the backend smoke tests accept.
- Overlay text remains readable on bright and dark webcam backgrounds.
- v4l2loopback virtual camera can show the overlayed preview in Zoom or Google Meet.
