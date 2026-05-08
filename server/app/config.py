from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "realtime-ksl-captioning"
    model_backend: Literal["mock", "huggingface"] = "mock"
    hf_model_id: str | None = None
    hf_model_revision: str = "main"
    hf_token: str | None = None
    model_device: str = "cuda:0"
    caption_auth_token: str | None = None

    frame_queue_size: int = Field(default=2, ge=1, le=30)
    max_frame_bytes: int = Field(default=2_000_000, ge=1)
    max_metadata_bytes: int = Field(default=16_384, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()

