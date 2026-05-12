from __future__ import annotations

from app.config import Settings

from .huggingface import HuggingFaceKslModel
from .interface import FrameForInference, KslModelAdapter
from .mock import MockKslModel


def build_model_adapter(settings: Settings) -> KslModelAdapter:
    if settings.model_backend == "mock":
        return MockKslModel()
    if settings.model_backend == "huggingface":
        return HuggingFaceKslModel(settings)
    raise ValueError(f"Unsupported MODEL_BACKEND: {settings.model_backend}")


__all__ = [
    "FrameForInference",
    "HuggingFaceKslModel",
    "KslModelAdapter",
    "MockKslModel",
    "build_model_adapter",
]
