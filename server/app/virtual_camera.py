import json
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import ValidationError

from app.config import Settings
from app.frame_packet import FramePacketError, parse_frame_packet
from app.image_utils import decode_jpeg_to_rgb
from app.schemas import ErrorEvent, StartMessage, StatusEvent


class VirtualCameraSink(Protocol):
    @property
    def device(self) -> str | None:
        ...

    def send(self, image_rgb: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...


VirtualCameraSinkFactory = Callable[[int, int, float, str | None], VirtualCameraSink]


class PyVirtualCameraSink:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        device: str | None = None,
    ) -> None:
        _validate_virtual_camera_device(device)

        try:
            import pyvirtualcam
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyvirtualcam is not installed. Install server requirements and configure v4l2loopback."
            ) from exc

        self.width = width
        self.height = height
        self._device = device
        self._camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
            device=device,
        )

    @property
    def device(self) -> str | None:
        return getattr(self._camera, "device", self._device)

    def send(self, image_rgb: np.ndarray) -> None:
        if image_rgb.shape[:2] != (self.height, self.width):
            image_rgb = np.asarray(
                Image.fromarray(image_rgb).resize((self.width, self.height))
            )
        self._camera.send(np.ascontiguousarray(image_rgb))

    def close(self) -> None:
        self._camera.close()


async def handle_virtual_camera_socket(
    websocket: WebSocket,
    *,
    session_id: str,
    settings: Settings,
    sink_factory: VirtualCameraSinkFactory | None = None,
) -> None:
    if not _is_authorized(websocket, settings):
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    await _send_status(websocket, session_id, "connected", {})

    try:
        start_message = await _receive_start_message(websocket)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await _send_error(websocket, session_id, "invalid_start", str(exc))
        await websocket.close(code=1003, reason="Invalid start message")
        return

    factory = sink_factory or PyVirtualCameraSink
    device = websocket.query_params.get("device") or None
    sink: VirtualCameraSink | None = None
    frames_sent = 0

    try:
        sink = factory(
            start_message.width,
            start_message.height,
            start_message.fps,
            device,
        )
    except Exception as exc:
        await _send_error(websocket, session_id, "virtual_camera_unavailable", str(exc))
        await websocket.close(code=1011, reason="Virtual camera unavailable")
        return

    await _send_status(
        websocket,
        session_id,
        "virtual_camera_started",
        {
            "width": start_message.width,
            "height": start_message.height,
            "fps": start_message.fps,
            "device": sink.device or device,
        },
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                frame_packet = await _parse_virtual_frame(
                    websocket,
                    session_id,
                    settings,
                    message["bytes"],
                )
                if frame_packet is None:
                    continue
                try:
                    sink.send(decode_jpeg_to_rgb(frame_packet.image_bytes))
                except Exception as exc:
                    await _send_error(websocket, session_id, "virtual_frame_failed", str(exc))
                    continue
                frames_sent += 1
                if frames_sent == 1:
                    await _send_status(
                        websocket,
                        session_id,
                        "virtual_frame_streaming",
                        {"frames_sent": frames_sent},
                    )
            elif message.get("text") is not None:
                should_stop = await _handle_text_message(
                    websocket,
                    session_id,
                    message["text"],
                    frames_sent,
                )
                if should_stop:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if sink is not None:
            with suppress(Exception):
                sink.close()


async def _receive_start_message(websocket: WebSocket) -> StartMessage:
    raw_message = await websocket.receive_text()
    try:
        data = json.loads(raw_message)
        return StartMessage.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Expected start JSON message: {exc}") from exc


async def _parse_virtual_frame(
    websocket: WebSocket,
    session_id: str,
    settings: Settings,
    payload: bytes,
):
    try:
        return parse_frame_packet(
            payload,
            max_metadata_bytes=settings.max_metadata_bytes,
            max_frame_bytes=settings.max_frame_bytes,
        )
    except FramePacketError as exc:
        await _send_error(websocket, session_id, "invalid_virtual_frame", str(exc))
        return None


async def _handle_text_message(
    websocket: WebSocket,
    session_id: str,
    raw_message: str,
    frames_sent: int,
) -> bool:
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        await _send_error(websocket, session_id, "invalid_message", "Text messages must be JSON.")
        return False

    message_type = message.get("type")
    if message_type == "stop":
        await _send_status(
            websocket,
            session_id,
            "virtual_camera_stopping",
            {"frames_sent": frames_sent},
        )
        return True
    if message_type == "ping":
        await _send_status(websocket, session_id, "pong", {})
        return False

    await _send_error(websocket, session_id, "unsupported_message", f"Unsupported message type: {message_type}")
    return False


async def _send_status(websocket: WebSocket, session_id: str, status: str, detail: dict) -> None:
    await websocket.send_json(StatusEvent(session_id=session_id, status=status, detail=detail).model_dump())


async def _send_error(websocket: WebSocket, session_id: str | None, code: str, message: str) -> None:
    await websocket.send_json(ErrorEvent(session_id=session_id, code=code, message=message).model_dump())


def _is_authorized(websocket: WebSocket, settings: Settings) -> bool:
    if not settings.caption_auth_token:
        return True

    query_token = websocket.query_params.get("token")
    if query_token == settings.caption_auth_token:
        return True

    authorization = websocket.headers.get("authorization", "")
    return authorization == f"Bearer {settings.caption_auth_token}"


def _validate_virtual_camera_device(device: str | None) -> None:
    if not device:
        return

    device_path = Path(device)
    if not device_path.exists():
        raise RuntimeError(
            f"Virtual camera device {device} does not exist. "
            "Run ./scripts/setup_virtual_camera_ubuntu.sh, then verify it with "
            f"'ls -l {device}'."
        )

    if not device_path.is_char_device():
        raise RuntimeError(f"{device} exists but is not a video device.")

    if not os.access(device_path, os.R_OK | os.W_OK):
        raise RuntimeError(
            f"Current user cannot read/write {device}. "
            "Confirm the user is in the video group, then log out and back in."
        )
