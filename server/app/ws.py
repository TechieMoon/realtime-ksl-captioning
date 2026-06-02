import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.backpressure import DropOldestQueue
from app.config import Settings
from app.frame_packet import FramePacket, FramePacketError, parse_frame_packet
from app.image_utils import decode_jpeg_to_rgb
from app.models.interface import FrameForInference, KslModelAdapter
from app.schemas import (
    CaptionEvent,
    ErrorEvent,
    SegmentEndMessage,
    SegmentStartMessage,
    StartMessage,
    StatusEvent,
)


@dataclass(frozen=True)
class InferenceJob:
    frames: list[FramePacket]
    segment_id: str | None = None

    @property
    def frame_id(self) -> int:
        return self.frames[-1].metadata.frame_id


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
    await _send_status(
        websocket,
        session_id,
        "connected",
        {"model_backend": settings.model_backend},
    )

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

    frame_queue: DropOldestQueue[InferenceJob] = DropOldestQueue(maxsize=settings.frame_queue_size)
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

    active_segment_id: str | None = None
    active_segment_frames: list[FramePacket] = []

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                frame_packet = await _parse_binary_frame(websocket, session_id, settings, message["bytes"])
                if frame_packet is None:
                    continue
                if active_segment_id is not None:
                    if len(active_segment_frames) >= settings.max_segment_frames:
                        await _send_error(
                            websocket,
                            session_id,
                            "segment_too_large",
                            f"Segment frame limit exceeded: {settings.max_segment_frames}",
                        )
                        continue
                    active_segment_frames.append(frame_packet)
                else:
                    await _enqueue_inference_job(
                        websocket,
                        session_id,
                        frame_queue,
                        InferenceJob(frames=[frame_packet]),
                    )
            elif message.get("text") is not None:
                should_stop, segment_update = await _handle_text_message(
                    websocket,
                    session_id,
                    message["text"],
                    active_segment_id,
                    len(active_segment_frames),
                )
                if should_stop:
                    break
                if segment_update is None:
                    continue
                action, segment_id = segment_update
                if action == "start":
                    active_segment_id = segment_id
                    active_segment_frames = []
                elif action == "end":
                    if not active_segment_frames:
                        await _send_error(websocket, session_id, "empty_segment", "Segment has no frames.")
                        active_segment_id = None
                        active_segment_frames = []
                        continue
                    await _enqueue_inference_job(
                        websocket,
                        session_id,
                        frame_queue,
                        InferenceJob(frames=active_segment_frames.copy(), segment_id=segment_id),
                    )
                    await _send_status(
                        websocket,
                        session_id,
                        "segment_queued",
                        {"segment_id": segment_id, "frames": len(active_segment_frames)},
                    )
                    active_segment_id = None
                    active_segment_frames = []
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


async def _parse_binary_frame(
    websocket: WebSocket,
    session_id: str,
    settings: Settings,
    payload: bytes,
) -> FramePacket | None:
    try:
        return parse_frame_packet(
            payload,
            max_metadata_bytes=settings.max_metadata_bytes,
            max_frame_bytes=settings.max_frame_bytes,
        )
    except FramePacketError as exc:
        await _send_error(websocket, session_id, "invalid_frame", str(exc))
        return None


async def _enqueue_inference_job(
    websocket: WebSocket,
    session_id: str,
    frame_queue: DropOldestQueue[InferenceJob],
    job: InferenceJob,
) -> None:
    dropped = await frame_queue.put(job)
    if dropped is not None:
        await _send_status(
            websocket,
            session_id,
            "inference_job_dropped",
            {
                "dropped_frame_id": dropped.frame_id,
                "dropped_segment_id": dropped.segment_id,
                "kept_frame_id": job.frame_id,
                "kept_segment_id": job.segment_id,
            },
        )


async def _handle_text_message(
    websocket: WebSocket,
    session_id: str,
    raw_message: str,
    active_segment_id: str | None,
    active_segment_frame_count: int,
) -> tuple[bool, tuple[str, str] | None]:
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError:
        await _send_error(websocket, session_id, "invalid_message", "Text messages must be JSON.")
        return False, None

    message_type = message.get("type")
    if message_type == "stop":
        await _send_status(websocket, session_id, "session_stopping", {})
        return True, None
    if message_type == "ping":
        await _send_status(websocket, session_id, "pong", {})
        return False, None
    if message_type == "segment_start":
        try:
            segment = SegmentStartMessage.model_validate(message)
        except ValidationError as exc:
            await _send_error(websocket, session_id, "invalid_segment_start", str(exc))
            return False, None
        if active_segment_id is not None:
            await _send_error(websocket, session_id, "segment_already_active", active_segment_id)
            return False, None
        await _send_status(websocket, session_id, "segment_started", {"segment_id": segment.segment_id})
        return False, ("start", segment.segment_id)
    if message_type == "segment_end":
        try:
            segment = SegmentEndMessage.model_validate(message)
        except ValidationError as exc:
            await _send_error(websocket, session_id, "invalid_segment_end", str(exc))
            return False, None
        if active_segment_id is None:
            await _send_error(websocket, session_id, "no_active_segment", segment.segment_id)
            return False, None
        if segment.segment_id != active_segment_id:
            await _send_error(websocket, session_id, "segment_id_mismatch", active_segment_id)
            return False, None
        await _send_status(
            websocket,
            session_id,
            "segment_ending",
            {"segment_id": segment.segment_id, "frames": active_segment_frame_count},
        )
        return False, ("end", segment.segment_id)

    await _send_error(websocket, session_id, "unsupported_message", f"Unsupported message type: {message_type}")
    return False, None


async def _caption_worker(
    *,
    websocket: WebSocket,
    session_id: str,
    frame_queue: DropOldestQueue[InferenceJob],
    model: KslModelAdapter,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            job = await asyncio.wait_for(frame_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        start_time = time.perf_counter()
        try:
            frames = [
                FrameForInference(
                    frame_id=frame_packet.metadata.frame_id,
                    timestamp_ms=frame_packet.metadata.timestamp_ms,
                    image_rgb=decode_jpeg_to_rgb(frame_packet.image_bytes),
                )
                for frame_packet in job.frames
            ]
            predictions = await model.predict(frames)
            latency_ms = (time.perf_counter() - start_time) * 1000
            if job.segment_id:
                predictions = predictions[-1:]
            for prediction in predictions:
                event = CaptionEvent(
                    session_id=session_id,
                    frame_id=prediction.frame_id,
                    text=prediction.text,
                    words=prediction.words,
                    is_final=prediction.is_final,
                    latency_ms=round(latency_ms, 2),
                    segment_id=job.segment_id,
                )
                await websocket.send_json(event.model_dump())
            if job.segment_id:
                await _send_status(
                    websocket,
                    session_id,
                    "segment_captioned",
                    {"segment_id": job.segment_id, "frames": len(job.frames), "latency_ms": round(latency_ms, 2)},
                )
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
