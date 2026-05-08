import asyncio
import json
import time
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.backpressure import DropOldestQueue
from app.config import Settings
from app.frame_packet import FramePacket, FramePacketError, parse_frame_packet
from app.image_utils import decode_jpeg_to_rgb
from app.models.interface import FrameForInference, KslModelAdapter
from app.schemas import CaptionEvent, ErrorEvent, StartMessage, StatusEvent


async def handle_caption_socket(
    websocket: WebSocket,
    *,
    session_id: str,
    settings: Settings,
    model: KslModelAdapter,
) -> None:
    if not _is_authorized(websocket, settings):
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    await _send_status(websocket, session_id, "connected", {"model_backend": settings.model_backend})

    try:
        start_message = await _receive_start_message(websocket)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await _send_error(websocket, session_id, "invalid_start", str(exc))
        await websocket.close(code=1003, reason="Invalid start message")
        return

    await _send_status(
        websocket,
        session_id,
        "session_started",
        {
            "width": start_message.width,
            "height": start_message.height,
            "fps": start_message.fps,
            "format": start_message.format,
            "queue_size": settings.frame_queue_size,
        },
    )

    frame_queue: DropOldestQueue[FramePacket] = DropOldestQueue(maxsize=settings.frame_queue_size)
    stop_event = asyncio.Event()
    worker = asyncio.create_task(
        _caption_worker(
            websocket=websocket,
            session_id=session_id,
            frame_queue=frame_queue,
            model=model,
            stop_event=stop_event,
        )
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await _handle_binary_frame(websocket, session_id, settings, frame_queue, message["bytes"])
            elif message.get("text") is not None:
                should_stop = await _handle_text_message(websocket, session_id, message["text"])
                if should_stop:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


async def _receive_start_message(websocket: WebSocket) -> StartMessage:
    raw_message = await websocket.receive_text()
    try:
        data = json.loads(raw_message)
        return StartMessage.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Expected start JSON message: {exc}") from exc


async def _handle_binary_frame(
    websocket: WebSocket,
    session_id: str,
    settings: Settings,
    frame_queue: DropOldestQueue[FramePacket],
    payload: bytes,
) -> None:
    try:
        frame_packet = parse_frame_packet(
            payload,
            max_metadata_bytes=settings.max_metadata_bytes,
            max_frame_bytes=settings.max_frame_bytes,
        )
    except FramePacketError as exc:
        await _send_error(websocket, session_id, "invalid_frame", str(exc))
        return

    dropped = await frame_queue.put(frame_packet)
    if dropped is not None:
        await _send_status(
            websocket,
            session_id,
            "frame_dropped",
            {
                "dropped_frame_id": dropped.metadata.frame_id,
                "kept_frame_id": frame_packet.metadata.frame_id,
            },
        )


async def _handle_text_message(websocket: WebSocket, session_id: str, raw_message: str) -> bool:
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        await _send_error(websocket, session_id, "invalid_message", "Text messages must be JSON.")
        return False

    message_type = message.get("type")
    if message_type == "stop":
        await _send_status(websocket, session_id, "session_stopping", {})
        return True
    if message_type == "ping":
        await _send_status(websocket, session_id, "pong", {})
        return False

    await _send_error(websocket, session_id, "unsupported_message", f"Unsupported message type: {message_type}")
    return False


async def _caption_worker(
    *,
    websocket: WebSocket,
    session_id: str,
    frame_queue: DropOldestQueue[FramePacket],
    model: KslModelAdapter,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            frame_packet = await asyncio.wait_for(frame_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        start_time = time.perf_counter()
        try:
            image_rgb = decode_jpeg_to_rgb(frame_packet.image_bytes)
            frame = FrameForInference(
                frame_id=frame_packet.metadata.frame_id,
                timestamp_ms=frame_packet.metadata.timestamp_ms,
                image_rgb=image_rgb,
            )
            predictions = await model.predict([frame])
            latency_ms = (time.perf_counter() - start_time) * 1000
            for prediction in predictions:
                event = CaptionEvent(
                    session_id=session_id,
                    frame_id=prediction.frame_id,
                    text=prediction.text,
                    words=prediction.words,
                    is_final=prediction.is_final,
                    latency_ms=round(latency_ms, 2),
                )
                await websocket.send_json(event.model_dump())
        except Exception as exc:
            await _send_error(websocket, session_id, "inference_failed", str(exc))


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

