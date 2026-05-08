from typing import Any, Literal

from pydantic import BaseModel, Field


class StartMessage(BaseModel):
    type: Literal["start"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0, le=60)
    format: Literal["jpeg"] = "jpeg"
    client_name: str | None = None


class FrameMetadata(BaseModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    format: Literal["jpeg", "jpg"] = "jpeg"


class WordCaption(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class CaptionPrediction(BaseModel):
    frame_id: int = Field(ge=0)
    text: str
    words: list[WordCaption] = Field(default_factory=list)
    is_final: bool = True


class CaptionEvent(BaseModel):
    type: Literal["caption"] = "caption"
    session_id: str
    frame_id: int
    text: str
    words: list[WordCaption]
    is_final: bool
    latency_ms: float


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    session_id: str
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    session_id: str | None = None
    code: str
    message: str

