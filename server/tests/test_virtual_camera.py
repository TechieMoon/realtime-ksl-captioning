import json
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.frame_packet import build_frame_packet
from app.main import create_app


class RecordingVirtualCameraSink:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        device: str | None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._device = device
        self.frames: list[np.ndarray] = []
        self.closed = False

    @property
    def device(self) -> str | None:
        return self._device

    def send(self, image_rgb: np.ndarray) -> None:
        self.frames.append(image_rgb)

    def close(self) -> None:
        self.closed = True


def test_virtual_camera_websocket_streams_frames_to_sink() -> None:
    sinks: list[RecordingVirtualCameraSink] = []

    def sink_factory(
        width: int,
        height: int,
        fps: float,
        device: str | None,
    ) -> RecordingVirtualCameraSink:
        sink = RecordingVirtualCameraSink(width, height, fps, device)
        sinks.append(sink)
        return sink

    app = create_app(
        Settings(model_backend="mock"),
        virtual_camera_sink_factory=sink_factory,
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/virtual-camera?session_id=virtual-test&device=/dev/video20"
        ) as websocket:
            connected = websocket.receive_json()
            websocket.send_text(
                json.dumps(
                    {
                        "type": "start",
                        "width": 32,
                        "height": 24,
                        "fps": 8,
                        "format": "jpeg",
                    }
                )
            )
            started = websocket.receive_json()
            websocket.send_bytes(
                build_frame_packet(
                    {
                        "frame_id": 0,
                        "timestamp_ms": 0,
                        "width": 32,
                        "height": 24,
                        "format": "jpeg",
                    },
                    _jpeg_bytes(),
                )
            )
            streaming = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "stop"}))
            stopping = websocket.receive_json()

    assert connected["status"] == "connected"
    assert started["status"] == "virtual_camera_started"
    assert started["detail"]["device"] == "/dev/video20"
    assert streaming["status"] == "virtual_frame_streaming"
    assert stopping["status"] == "virtual_camera_stopping"
    assert len(sinks) == 1
    assert sinks[0].width == 32
    assert sinks[0].height == 24
    assert sinks[0].fps == 8
    assert sinks[0].frames[0].shape == (24, 32, 3)
    assert sinks[0].closed


def test_virtual_camera_websocket_reports_sink_failure() -> None:
    def sink_factory(
        width: int,
        height: int,
        fps: float,
        device: str | None,
    ) -> RecordingVirtualCameraSink:
        raise RuntimeError("no virtual camera")

    app = create_app(
        Settings(model_backend="mock"),
        virtual_camera_sink_factory=sink_factory,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/virtual-camera?session_id=virtual-test") as websocket:
            websocket.receive_json()
            websocket.send_text(
                json.dumps(
                    {
                        "type": "start",
                        "width": 32,
                        "height": 24,
                        "fps": 8,
                        "format": "jpeg",
                    }
                )
            )
            error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "virtual_camera_unavailable"
    assert "no virtual camera" in error["message"]


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
