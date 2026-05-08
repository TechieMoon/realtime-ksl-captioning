import asyncio
import json
from io import BytesIO

from PIL import Image, ImageDraw
import websockets

from app.frame_packet import build_frame_packet


async def main() -> None:
    uri = "ws://localhost:8000/ws/captions?session_id=smoke"
    async with websockets.connect(uri, max_size=4_000_000) as websocket:
        print(await websocket.recv())
        await websocket.send(json.dumps({"type": "start", "width": 320, "height": 180, "fps": 8, "format": "jpeg"}))
        print(await websocket.recv())

        for frame_id in range(5):
            await websocket.send(build_frame_packet(_metadata(frame_id), _jpeg(frame_id)))
            print(await websocket.recv())
            await asyncio.sleep(0.12)

        await websocket.send(json.dumps({"type": "stop"}))
        print(await websocket.recv())


def _metadata(frame_id: int) -> dict:
    return {"frame_id": frame_id, "timestamp_ms": frame_id * 125, "width": 320, "height": 180, "format": "jpeg"}


def _jpeg(frame_id: int) -> bytes:
    image = Image.new("RGB", (320, 180), color=(30, 35, 45))
    draw = ImageDraw.Draw(image)
    draw.text((20, 70), f"KSL frame {frame_id}", fill=(245, 245, 245))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


if __name__ == "__main__":
    asyncio.run(main())

