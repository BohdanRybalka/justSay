"""Application-level routes: health check and graceful shutdown."""

import signal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import __version__
from app.core.config import settings
from app.core.types import ProviderMode

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    stt_mode: ProviderMode


class ShutdownResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        stt_mode=settings.stt.mode,
    )


@router.post("/shutdown", response_model=ShutdownResponse, status_code=202)
async def request_shutdown() -> ShutdownResponse:
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="Shutdown requires a configured API token")
    _raise_stop_signal()
    return ShutdownResponse(status="stopping")


def _raise_stop_signal() -> None:
    signal.raise_signal(signal.SIGTERM)
