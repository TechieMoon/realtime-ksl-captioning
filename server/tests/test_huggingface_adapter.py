from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.models.huggingface import HuggingFaceKslModel, _import_repo_module
from app.models.interface import FrameForInference


def test_import_repo_module_bypasses_broken_package_init(tmp_path: Path) -> None:
    package_dir = tmp_path / "model"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("from .missing import Missing\n", encoding="utf-8")
    (package_dir / "helper.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (package_dir / "predict_word_classifier.py").write_text(
        "from .helper import VALUE\n",
        encoding="utf-8",
    )

    module = _import_repo_module(tmp_path, "model.predict_word_classifier")

    assert module.VALUE == "ok"


@pytest.mark.asyncio
async def test_huggingface_adapter_loads_local_snapshot_contract(tmp_path: Path) -> None:
    inference_file = tmp_path / "inference.py"
    inference_file.write_text(
        """
def load_model(model_dir, device):
    return {"model_dir": model_dir, "device": device}

def predict(model, frames, timestamps_ms):
    return [
        {
            "text": "안녕하세요",
            "words": [{"text": "안녕하세요", "confidence": 0.87, "start_ms": 10, "end_ms": 410}],
            "is_final": True,
        }
    ]
""".strip(),
        encoding="utf-8",
    )

    adapter = HuggingFaceKslModel(
        Settings(model_backend="huggingface", hf_model_id=str(tmp_path), model_device="cpu")
    )
    await adapter.load()

    predictions = await adapter.predict(
        [FrameForInference(frame_id=3, timestamp_ms=10, image_rgb=np.zeros((8, 8, 3), dtype=np.uint8))]
    )

    assert adapter.ready is True
    assert predictions[0].frame_id == 3
    assert predictions[0].text == "안녕하세요"
    assert predictions[0].words[0].confidence == 0.87
