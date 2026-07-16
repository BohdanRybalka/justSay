"""STT Local mode readiness checks — package detection, GPU, pip install."""

import asyncio
import logging
import subprocess
import sys
from collections.abc import AsyncIterator

from pydantic import BaseModel

from app.core.utils import sse_event
from app.stt.config import STTSettings
from app.stt.local_factory import is_macos_arm64

log = logging.getLogger(__name__)

_install_lock = asyncio.Lock()


class LocalSttStatus(BaseModel):
    package_installed: bool = False
    model_loaded: bool = False
    model_name: str = ""
    model_ram_mb: int | None = None
    gpu_available: bool = False
    gpu_name: str | None = None
    # "apple" on macOS arm64, else the app.core.gpu_probe vendor value
    # ("nvidia"/"amd"/"intel"/"none"). Populated even when gpu_available is
    # False — AMD/Intel are detected but not yet STT-accelerated.
    gpu_vendor: str = "none"
    device: str = "cpu"
    compute_type: str = "int8"
    last_error: str | None = None


def check_status(stt_settings: STTSettings) -> LocalSttStatus:
    """Check local STT readiness: package installed + load state + GPU + last error."""
    installed = _check_package_installed()
    gpu_available, gpu_name, gpu_vendor = _detect_gpu()

    if is_macos_arm64():
        device = "mlx"
        # Informational label — actual dtype is controlled inside mlx-whisper.
        # Accurate for the project default large-v3-turbo and other large
        # variants; smaller MLX checkpoints ship as float16.
        compute_type = "bfloat16"
    else:
        device = stt_settings.whisper_device
        if device == "auto":
            device = "cuda" if gpu_available else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

    # is_model_loaded reads the cached provider; safe even if the package is missing.
    from app.stt import get_local_load_error, is_model_loaded

    return LocalSttStatus(
        package_installed=installed,
        model_loaded=is_model_loaded() if installed else False,
        model_name=stt_settings.whisper_model_size,
        model_ram_mb=_estimate_model_ram_mb() if is_model_loaded() else None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vendor=gpu_vendor,
        device=device,
        compute_type=compute_type,
        last_error=get_local_load_error(stt_settings),
    )


def _estimate_model_ram_mb() -> int | None:
    """Approximate the backend RSS-delta consumed by the loaded whisper model.

    Returns the current process RSS in MB — coarse but informative; the user
    sees "the backend is holding ~700 MB" rather than no number at all.
    """
    try:
        import os

        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss
        return rss // (1024 * 1024)
    except Exception:  # psutil missing on a stripped install
        return None


def _local_extras() -> str:
    """Return the `pip install .[<extra>]` extras name for the current platform."""
    return "local-mac" if is_macos_arm64() else "local"


def _check_package_installed() -> bool:
    """Check if the platform-appropriate local STT package is importable."""
    try:
        if is_macos_arm64():
            import mlx_whisper  # noqa: F401
        else:
            import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


async def install_local_packages() -> AsyncIterator[str]:
    """Install local STT dependencies via pip with SSE progress.

    Runs: pip install .[local] from the backend directory.
    Yields SSE-formatted strings.
    """
    if _install_lock.locked():
        yield sse_event("error", {"status": "error", "error": "Installation already in progress"})
        return

    # pip install is impossible from a PyInstaller-frozen sidecar: there's no
    # pyproject.toml, no editable source tree, and sys.executable points at
    # the frozen interpreter itself. Surface a clear error instead of running
    # pip from `_MEIPASS` with surprising side effects.
    if getattr(sys, "frozen", False):
        yield sse_event("error", {
            "status": "error",
            "error": (
                "Local STT install is not supported in the packaged build. "
                "Install JustSay from source if you need Local mode on this OS."
            ),
        })
        return

    # Already installed?
    if _check_package_installed():
        yield sse_event("done", {"status": "already_installed"})
        return

    async with _install_lock:
        yield sse_event("progress", {"status": "Installing local dependencies..."})

        try:
            exit_code, output = await asyncio.to_thread(_run_pip_install)
            if exit_code == 0:
                yield sse_event("done", {"status": "success"})
            else:
                yield sse_event("error", {"status": "error", "error": output[-500:] if output else "pip install failed"})
        except Exception as e:
            log.warning("pip install failed: %s", e)
            yield sse_event("error", {"status": "error", "error": str(e)})


def _run_pip_install() -> tuple[int, str]:
    """Run pip install .[<extras>] synchronously. Returns (exit_code, output)."""
    backend_dir = _get_backend_dir()
    extras = _local_extras()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", f".[{extras}]"],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
        timeout=300,  # 5 min max
    )

    output = result.stdout + result.stderr
    log.info("pip install exit code: %d", result.returncode)
    if result.returncode != 0:
        log.warning("pip install output: %s", output[-1000:])

    return result.returncode, output


def _get_backend_dir():
    """Get the backend project directory (where pyproject.toml lives)."""
    from pathlib import Path

    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: assume backend/ is two levels up from app/stt/
    return Path(__file__).resolve().parent.parent.parent


def _detect_gpu() -> tuple[bool, str | None, str]:
    """Detect GPU availability, a human-readable device name, and vendor.

    On macOS arm64 returns the Apple-Silicon/Metal label without importing
    torch or probing hardware — the MLX path is the accelerator here.
    Everywhere else, delegates to `app.core.gpu_probe.probe_gpu()`.
    `gpu_available` stays true only for the actually-accelerated path
    (faster-whisper has no AMD/Intel backend), but `gpu_name`/vendor are
    populated for AMD/Intel too so the UI can show "GPU detected, not yet
    accelerated" instead of nothing.

    Returns (available, device_name_or_none, vendor).
    """
    if is_macos_arm64():
        return True, "Apple Silicon (MLX/Metal)", "apple"

    from app.core.gpu_probe import GpuVendor, probe_gpu

    result = probe_gpu()
    available = result.vendor == GpuVendor.NVIDIA
    return available, result.name, result.vendor.value
