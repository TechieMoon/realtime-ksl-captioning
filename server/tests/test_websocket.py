import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.frame_packet import build_frame_packet
from app.main import create_app


def test_caption_websocket_receives_frame_and_returns_caption() -> None:
    app = create_app(Settings(model_backend="mock"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/captions?session_id=test-session") as websocket:
            connected = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "start", "width": 32, "height": 24, "fps": 8, "format": "jpeg"}))
            started = websocket.receive_json()

            websocket.send_bytes(
                build_frame_packet(
                    {"frame_id": 0, "timestamp_ms": 0, "width": 32, "height": 24, "format": "jpeg"},
                    _jpeg_bytes(),
                )
            )
            caption = websocket.receive_json()

    assert connected["status"] == "connected"
    assert started["status"] == "session_started"
    assert caption["type"] == "caption"
    assert caption["session_id"] == "test-session"
    assert caption["frame_id"] == 0
    assert caption["words"][0]["text"]


def test_caption_websocket_requires_start_message() -> None:
    app = create_app(Settings(model_backend="mock"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/captions?session_id=test-session") as websocket:
            websocket.receive_json()
            websocket.send_text(json.dumps({"type": "wrong"}))
            error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "invalid_start"


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

