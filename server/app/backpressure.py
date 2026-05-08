import asyncio
from typing import Generic, TypeVar


T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Bounded async queue that keeps the newest frames when producers are faster."""

    def __init__(self, maxsize: int):
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: T) -> T | None:
        dropped: T | None = None
        if self._queue.full():
            dropped = self._queue.get_nowait()
        await self._queue.put(item)
        return dropped

    async def get(self) -> T:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()

