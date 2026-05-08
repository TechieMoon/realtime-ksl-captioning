import numpy as np
import pytest

from app.models.interface import FrameForInference
from app.models.mock import MockKslModel


@pytest.mark.asyncio
async def test_mock_model_returns_caption_schema() -> None:
    model = MockKslModel()
    await model.load()

    predictions = await model.predict(
        [FrameForInference(frame_id=0, timestamp_ms=0, image_rgb=np.zeros((4, 4, 3), dtype=np.uint8))]
    )

    assert predictions[0].frame_id == 0
    assert predictions[0].text
    assert predictions[0].words[0].confidence > 0
    assert predictions[0].is_final is True

