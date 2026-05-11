"""STT Local mode readiness checks — package detection, GPU, pip install."""

import asyncio
import logging
import subprocess
import sys
from collections.abc import AsyncIterator

from pydantic import BaseModel

from app.core.utils import sse_event
from app.stt.config import STTSettings

log = logging.getLogger(__name__)

_install_lock = asyncio.Lock()


class LocalSttStatus(BaseModel):
    package_installed: bool = False
    model_loaded: bool = False
    model_name: str = ""
    model_ram_mb: int | None = None
    gpu_available: bool = False
    gpu_name: str | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    last_error: str | None = None


def check_status(stt_settings: STTSettings) -> LocalSttStatus:
    """Check local STT readiness: package installed + load state + GPU + last error."""
    installed = _check_package_installed()
    gpu_available, gpu_name = _detect_gpu()

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


def _check_package_installed() -> bool:
    """Check if faster-whisper is importable."""
    try:
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
    """Run pip install .[local] synchronously. Returns (exit_code, output)."""
    backend_dir = _get_backend_dir()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", ".[local]"],
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


def _detect_gpu() -> tuple[bool, str | None]:
    """Detect CUDA GPU availability and name.

    Returns (available, device_name_or_none).
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
    except ImportError:
        pass
    except Exception as e:
        log.warning("GPU detection failed: %s", e)

    return False, None
