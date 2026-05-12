from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.schemas import CaptionPrediction


@dataclass(frozen=True, slots=True)
class FrameForInference:
    frame_id: int
    timestamp_ms: int | None
    image_rgb: np.ndarray


class KslModelAdapter(Protocol):
    name: str

    @property
    def ready(self) -> bool:
        ...

    async def load(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def predict(self, frames: list[FrameForInference]) -> list[CaptionPrediction]:
        ...
