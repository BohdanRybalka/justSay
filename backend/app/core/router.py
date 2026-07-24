"""Core routes: health check, config aggregation, resource monitoring."""

import asyncio
import os
import signal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import __version__
from app.core.config import settings
from app.core.schemas import ConfigResponse, HealthResponse, ShutdownResponse
from app.core.utils import bytes_to_gb, bytes_to_mb
from app.stt import get_provider as get_stt_provider

router = APIRouter()


_proc_cache = None


def _get_proc():
    global _proc_cache
    if _proc_cache is None:
        import psutil

        _proc_cache = psutil.Process(os.getpid())
        _proc_cache.cpu_percent(None)
    return _proc_cache


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


class GpuInfo(BaseModel):
    name: str
    vendor: str
    vram_total_mb: int
    vram_used_mb: int | None
    vram_free_mb: int | None


class ResourceInfo(BaseModel):
    cpu_cores: int
    cpu_threads: int
    cpu_percent_total: float
    cpu_percent_process: float
    ram_total_mb: int
    ram_used_mb: int
    ram_available_mb: int
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    pid_ram_mb: int
    pid_ram_gb: float
    gpu: GpuInfo | None = None


@router.get("/resources", response_model=ResourceInfo)
async def get_resources():
    """System resource usage: CPU, RAM, GPU (if available)."""
    return await asyncio.to_thread(_collect_resources)


def _collect_resources() -> ResourceInfo:
    import psutil

    proc = _get_proc()
    vm = psutil.virtual_memory()
    threads = psutil.cpu_count(logical=True) or 1

    rss_bytes = proc.memory_info().rss
    proc_cpu_raw = proc.cpu_percent(None)
    proc_cpu_normalised = min(proc_cpu_raw / threads, 100.0)

    gpu = _get_gpu_info()

    return ResourceInfo(
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=threads,
        cpu_percent_total=psutil.cpu_percent(interval=None),
        cpu_percent_process=round(proc_cpu_normalised, 1),
        ram_total_mb=bytes_to_mb(vm.total),
        ram_used_mb=bytes_to_mb(vm.used),
        ram_available_mb=bytes_to_mb(vm.available),
        ram_total_gb=bytes_to_gb(vm.total),
        ram_used_gb=bytes_to_gb(vm.used),
        ram_available_gb=bytes_to_gb(vm.available),
        pid_ram_mb=bytes_to_mb(rss_bytes),
        pid_ram_gb=bytes_to_gb(rss_bytes),
        gpu=gpu,
    )


def _get_gpu_info() -> GpuInfo | None:
    """Populate the Resources tab's GPU row for any detected vendor, not just NVIDIA.

    `vram_used_mb`/`vram_free_mb` stay `None` for the AMD/Intel registry
    source, which has no live-usage reading (see `app.core.gpu_probe`).
    """
    from app.core.gpu_probe import GpuVendor, probe_gpu

    result = probe_gpu()
    if result.vendor == GpuVendor.NONE:
        return None

    return GpuInfo(
        name=result.name or "Unknown GPU",
        vendor=result.vendor.value,
        vram_total_mb=result.vram_total_mb or 0,
        vram_used_mb=result.vram_used_mb,
        vram_free_mb=result.vram_free_mb,
    )
