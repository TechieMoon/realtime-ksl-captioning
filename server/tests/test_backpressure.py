import pytest

from app.backpressure import DropOldestQueue


@pytest.mark.asyncio
async def test_drop_oldest_queue_keeps_newest_item() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(maxsize=1)

    first_drop = await queue.put(1)
    second_drop = await queue.put(2)

    assert first_drop is None
    assert second_drop == 1
    assert await queue.get() == 2

