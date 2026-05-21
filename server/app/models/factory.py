from app.config import Settings
from app.models.huggingface import HuggingFaceKslModel
from app.models.interface import KslModelAdapter
from app.models.mock import MockKslModel


def build_model_adapter(settings: Settings) -> KslModelAdapter:
    if settings.model_backend == "mock":
        return MockKslModel()
    if settings.model_backend == "huggingface":
        return HuggingFaceKslModel(settings)
    raise ValueError(f"Unsupported MODEL_BACKEND: {settings.model_backend}")

