import pytest

from app.frame_packet import FramePacketError, build_frame_packet, parse_frame_packet


def test_parse_frame_packet_round_trip() -> None:
    packet = build_frame_packet(
        {"frame_id": 7, "timestamp_ms": 1234, "width": 320, "height": 180, "format": "jpeg"},
        b"jpeg-bytes",
    )

    parsed = parse_frame_packet(packet, max_metadata_bytes=1024, max_frame_bytes=1024)

    assert parsed.metadata.frame_id == 7
    assert parsed.metadata.timestamp_ms == 1234
    assert parsed.image_bytes == b"jpeg-bytes"


def test_parse_frame_packet_rejects_missing_image_bytes() -> None:
    packet = build_frame_packet({"frame_id": 1, "format": "jpeg"}, b"")

    with pytest.raises(FramePacketError):
        parse_frame_packet(packet, max_metadata_bytes=1024, max_frame_bytes=1024)

