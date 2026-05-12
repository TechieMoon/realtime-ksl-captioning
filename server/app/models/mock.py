from __future__ import annotations

from app.schemas import CaptionPrediction, WordCaption

from .interface import FrameForInference


class MockKslModel:
    name = "mock"

    def __init__(self) -> None:
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def load(self) -> None:
        self._ready = True

    async def close(self) -> None:
        self._ready = False

    async def predict(self, frames: list[FrameForInference]) -> list[CaptionPrediction]:
        predictions: list[CaptionPrediction] = []
        for frame in frames:
            start_ms = frame.timestamp_ms or 0
            end_ms = start_ms + 500
            predictions.append(
                CaptionPrediction(
                    frame_id=frame.frame_id,
                    text="안녕하세요",
                    words=[
                        WordCaption(
                            text="안녕하세요",
                            confidence=0.92,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                    ],
                    is_final=True,
                )
            )
        return predictions
