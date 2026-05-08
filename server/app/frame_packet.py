import json
from dataclasses import dataclass

from pydantic import ValidationError

from app.schemas import FrameMetadata


class FramePacketError(ValueError):
    """Raised when a binary WebSocket frame packet is malformed."""


@dataclass(frozen=True)
class FramePacket:
    metadata: FrameMetadata
    image_bytes: bytes


def parse_frame_packet(
    payload: bytes,
    *,
    max_metadata_bytes: int,
    max_frame_bytes: int,
) -> FramePacket:
    """Parse a binary packet: 4-byte metadata length, JSON metadata, JPEG bytes."""

    if len(payload) < 5:
        raise FramePacketError("Frame packet must contain a 4-byte metadata length and JPEG bytes.")

    metadata_length = int.from_bytes(payload[:4], byteorder="big", signed=False)
    if metadata_length <= 0:
        raise FramePacketError("Frame packet metadata length must be positive.")
    if metadata_length > max_metadata_bytes:
        raise FramePacketError("Frame packet metadata is too large.")

    metadata_end = 4 + metadata_length
    if len(payload) <= metadata_end:
        raise FramePacketError("Frame packet is missing JPEG bytes.")

    image_bytes = payload[metadata_end:]
    if len(image_bytes) > max_frame_bytes:
        raise FramePacketError("Frame JPEG payload is too large.")

    try:
        raw_metadata = json.loads(payload[4:metadata_end].decode("utf-8"))
        metadata = FrameMetadata.model_validate(raw_metadata)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise FramePacketError(f"Invalid frame metadata: {exc}") from exc

    return FramePacket(metadata=metadata, image_bytes=image_bytes)


def build_frame_packet(metadata: dict, image_bytes: bytes) -> bytes:
    """Build a binary packet for tests and smoke clients."""

    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return len(metadata_bytes).to_bytes(4, byteorder="big", signed=False) + metadata_bytes + image_bytes

