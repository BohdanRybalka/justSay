"""Core routes: health check, config aggregation, graceful shutdown."""

import signal

from fastapi import APIRouter, HTTPException

from app import __version__
from app.core.config import settings
from app.core.schemas import ConfigResponse, HealthResponse, ShutdownResponse
from app.stt import get_provider as get_stt_provider

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        stt_mode=settings.stt.mode,
        llm_mode=settings.llm.mode,
    )


@router.post("/shutdown", response_model=ShutdownResponse, status_code=202)
async def request_shutdown() -> ShutdownResponse:
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="Shutdown requires a configured API token")
    _raise_stop_signal()
    return ShutdownResponse(status="stopping")


def _raise_stop_signal() -> None:
    signal.raise_signal(signal.SIGTERM)


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    stt = get_stt_provider(settings.stt.mode, settings.stt)
    return ConfigResponse(
        stt_mode=settings.stt.mode,
        llm_mode=settings.llm.mode,
        stt_model=stt.model_name,
    )

