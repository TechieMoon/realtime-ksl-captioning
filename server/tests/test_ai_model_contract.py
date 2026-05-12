import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.models.huggingface import HuggingFaceKslModel
from app.models.interface import FrameForInference


def _load_repo_inference_module():
    repo_root = Path(__file__).resolve().parents[2]
    inference_file = repo_root / "ai_model" / "inference.py"
    spec = importlib.util.spec_from_file_location("local_ai_model_inference", inference_file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_ai_model_inference"] = module
    spec.loader.exec_module(module)
    return module


def test_ai_model_repo_contract_returns_one_prediction_per_frame() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "ai_model"
    module = _load_repo_inference_module()

    model = module.load_model(str(model_dir), "cpu")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    predictions = module.predict(model, [frame], [1200])

    assert len(predictions) == 1
    assert set(predictions[0]) == {"text", "words", "is_final"}
    assert isinstance(predictions[0]["text"], str)
    assert isinstance(predictions[0]["words"], list)
    assert isinstance(predictions[0]["is_final"], bool)


@pytest.mark.asyncio
async def test_huggingface_adapter_loads_checked_in_ai_model_repo() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "ai_model"
    adapter = HuggingFaceKslModel(
        Settings(model_backend="huggingface", hf_model_id=str(model_dir), model_device="cpu")
    )

    await adapter.load()
    predictions = await adapter.predict(
        [FrameForInference(frame_id=7, timestamp_ms=1200, image_rgb=np.zeros((32, 32, 3), dtype=np.uint8))]
    )

    assert adapter.ready is True
    assert predictions[0].frame_id == 7
    assert predictions[0].text == ""
    assert predictions[0].words == []
