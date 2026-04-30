"""Core routes: health check, config aggregation, resource monitoring."""

import os
import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.core.config import settings
from app.core.schemas import HealthResponse, ConfigResponse
from app.stt import get_stt_provider
from app.llm import get_llm_provider

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        stt_mode=settings.stt.mode,
        llm_mode=settings.llm.mode,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    stt = get_stt_provider(settings.stt)
    llm = get_llm_provider(settings.llm)
    return ConfigResponse(
        stt_mode=settings.stt.mode,
        llm_mode=settings.llm.mode,
        stt_model=stt.model_name,
        llm_model=llm.model_name,
    )


class GpuInfo(BaseModel):
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int


class ResourceInfo(BaseModel):
    cpu_cores: int
    cpu_threads: int
    ram_total_mb: int
    ram_used_mb: int
    ram_available_mb: int
    pid_ram_mb: int
    gpu: GpuInfo | None = None


@router.get("/resources", response_model=ResourceInfo)
async def get_resources():
    """System resource usage: CPU, RAM, GPU (if available)."""
    return await asyncio.to_thread(_collect_resources)


def _collect_resources() -> ResourceInfo:
    import psutil

    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()

    gpu = _get_gpu_info()

    return ResourceInfo(
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=vm.total // (1024 * 1024),
        ram_used_mb=vm.used // (1024 * 1024),
        ram_available_mb=vm.available // (1024 * 1024),
        pid_ram_mb=proc.memory_info().rss // (1024 * 1024),
        gpu=gpu,
    )


def _get_gpu_info() -> GpuInfo | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        total = props.total_mem
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total - reserved

        return GpuInfo(
            name=props.name,
            vram_total_mb=total // (1024 * 1024),
            vram_used_mb=allocated // (1024 * 1024),
            vram_free_mb=free // (1024 * 1024),
        )
    except (ImportError, Exception):
        return None
