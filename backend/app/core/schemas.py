"""Shared API schemas used across modules."""

from pydantic import BaseModel

from app.core.types import ProviderMode


class HealthResponse(BaseModel):
    status: str
    version: str
    stt_mode: ProviderMode
    llm_mode: ProviderMode


class ShutdownResponse(BaseModel):
    status: str
