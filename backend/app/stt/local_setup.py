"""STT Local mode readiness checks — package detection, GPU, pip install."""

import asyncio
import json
import logging
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel

from app.core import tasks
from app.core.errors import ResourceUnavailableError
from app.core.types import ProviderMode
from app.core.utils import sse_event
from app.stt import local_whisper_cpp_cmd, routing
from app.stt.base import latched_load_error
from app.stt.config import STTSettings
from app.stt.local_factory import (
    LocalProviderKind,
    compute_type_for_device,
    get_local_provider_kind,
    is_accelerated_device,
    is_macos_arm64,
)

log = logging.getLogger(__name__)

_install_lock = asyncio.Lock()

_prewarm_lock = asyncio.Lock()
_prewarm_error: str | None = None

_active_load: tuple[object, asyncio.Task] | None = None

_READY_TIMEOUT = 300.0


def peek_active_load() -> asyncio.Task | None:
    """Read-only view of the in-flight model-load task, if any.

    Exists so `lifespan()`'s shutdown drain can cancel it: `_active_load` is
    not registered in `app.core.tasks`, so the drain cannot find it there.
    """
    return _active_load[1] if _active_load is not None else None


class LocalReadinessTimeoutError(ResourceUnavailableError):
    """Raised by `await_local_ready` when its bounded wait genuinely times out.

    Never raised for `ensure_local_ready`'s own early returns, which come back
    promptly. Inherits the base class's 503 and `resource_unavailable` code.
    """


class LocalSTTStatus(BaseModel):
    package_installed: bool = False
    model_loaded: bool = False
    model_name: str = ""
    model_ram_mb: int | None = None
    gpu_available: bool = False
    gpu_name: str | None = None
    gpu_vendor: str = "none"
    device: str = "cpu"
    compute_type: str = "int8"
    last_error: str | None = None


def check_status(stt_settings: STTSettings) -> LocalSTTStatus:
    """Check local STT readiness: package installed + load state + GPU + last error.

    The provider cache is read for the loaded state exactly once, so
    ``model_loaded`` and ``model_ram_mb`` cannot contradict each other.
    """
    installed = _check_package_installed()
    cuda_probe_available, gpu_name, gpu_vendor = _detect_gpu()

    if is_macos_arm64():
        kind = LocalProviderKind.WHISPER_CPP_SERVER
        device = "metal"
    else:
        from app.core.gpu_probe import GpuVendor

        kind = get_local_provider_kind(GpuVendor(gpu_vendor))
        if kind == LocalProviderKind.WHISPER_CPP_SERVER:
            device = "vulkan"
        else:
            device = stt_settings.whisper_device
            if device == "auto":
                device = "cuda" if cuda_probe_available else "cpu"

    compute_type = compute_type_for_device(device, kind)
    gpu_available = is_accelerated_device(device, kind)

    last_error = routing.get_local_load_error(stt_settings) or _prewarm_error
    model_is_loaded = routing.is_model_loaded() if installed else False

    return LocalSTTStatus(
        package_installed=installed,
        model_loaded=model_is_loaded,
        model_name=stt_settings.whisper_model_size,
        model_ram_mb=_estimate_model_ram_mb() if model_is_loaded else None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vendor=gpu_vendor,
        device=device,
        compute_type=compute_type,
        last_error=last_error,
    )


def maybe_prewarm_local(stt_settings: STTSettings) -> None:
    """Fire-and-forget. No-op unless ``stt_settings.mode`` is LOCAL.

    For explicit triggers only — startup calls `maybe_prewarm_local_at_startup`
    instead — and it resets the startup crash-loop counter to 0.
    """
    if stt_settings.mode != ProviderMode.LOCAL:
        return
    _write_consecutive_incomplete_prewarms(0)
    tasks.spawn_background_task(ensure_local_ready(stt_settings), name="local-stt-prewarm")


MAX_CONSECUTIVE_INCOMPLETE_PREWARMS = 2
_CRASH_GUARD_FILENAME = "prewarm_crash_guard.json"


def _crash_guard_path() -> Path:
    from app.core.app_paths import resolve_app_data_root

    return resolve_app_data_root() / _CRASH_GUARD_FILENAME


def _read_consecutive_incomplete_prewarms() -> int:
    """Fail-open: a missing or corrupt marker reads as 0 -- never let a
    corrupted file permanently block a legitimate prewarm attempt."""
    try:
        data = json.loads(_crash_guard_path().read_text())
        return int(data.get("consecutive_incomplete_prewarms", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _write_consecutive_incomplete_prewarms(n: int) -> None:
    path = _crash_guard_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"consecutive_incomplete_prewarms": n}))
    except OSError:
        log.warning("Could not persist prewarm crash-guard state at %s", path, exc_info=True)


def should_skip_prewarm(consecutive_incomplete_starts: int) -> bool:
    """Pure decision function -- unit tested directly, no filesystem
    involved."""
    return consecutive_incomplete_starts >= MAX_CONSECUTIVE_INCOMPLETE_PREWARMS


def maybe_prewarm_local_at_startup(stt_settings: STTSettings) -> None:
    """Startup-only entry point, called once from `app.main.lifespan`.

    Skips the prewarm after `MAX_CONSECUTIVE_INCOMPLETE_PREWARMS` process
    starts whose load never completed; a spawn that raises restores the count.
    """
    if stt_settings.mode != ProviderMode.LOCAL:
        return

    consecutive = _read_consecutive_incomplete_prewarms()
    if should_skip_prewarm(consecutive):
        log.warning(
            "Local STT prewarm skipped at startup: %d consecutive prewarm "
            "attempts did not complete before the process restarted (the "
            "backend likely crashed or hung during model load, e.g. "
            "out-of-memory). Falling back to the existing on-demand load on "
            "the first dictation request. Call POST /stt/local/prewarm "
            "after addressing the underlying issue (free RAM, a smaller "
            "model) to retry -- any explicit prewarm trigger resets this "
            "counter.",
            consecutive,
        )
        return

    _write_consecutive_incomplete_prewarms(consecutive + 1)
    prewarm = _prewarm_then_clear_crash_guard(stt_settings)
    try:
        tasks.spawn_background_task(prewarm, name="local-stt-prewarm-startup")
    except Exception:
        prewarm.close()
        _write_consecutive_incomplete_prewarms(consecutive)
        raise


async def _prewarm_then_clear_crash_guard(stt_settings: STTSettings) -> None:
    try:
        await ensure_local_ready(stt_settings)
    finally:
        _write_consecutive_incomplete_prewarms(0)


async def _run_get_model(provider) -> None:
    """Run one ``_get_model()`` attempt, swallowing and latching its failure.

    Runs as its own Task so it completes, and writes ``_prewarm_error``, even
    when every watcher is cancelled; releases the provider the cache moved past.
    """
    global _prewarm_error
    try:
        await asyncio.to_thread(provider._get_model)
    except Exception as e:
        _prewarm_error = latched_load_error(e)
    else:
        _prewarm_error = None
    finally:
        if routing.peek_local_provider() is not provider:
            try:
                provider.cleanup()
            except Exception:
                log.warning("Releasing the superseded local provider failed", exc_info=True)


async def ensure_local_ready(stt_settings: STTSettings) -> None:
    """Install if needed and load the Local STT model, serialized through
    ``_prewarm_lock``. The load runs as a shielded Task, so a cancelled caller
    neither stops it nor starts a second one; a caller whose provider the cache
    has moved past returns early and lets that provider be released.
    """
    global _prewarm_error, _active_load
    async with _prewarm_lock:
        if stt_settings.mode != ProviderMode.LOCAL:
            return

        provider = routing.get_provider(ProviderMode.LOCAL, stt_settings)
        if provider.is_loaded:
            _prewarm_error = None
            return

        if not _check_package_installed():
            if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_SERVER:
                _prewarm_error = local_whisper_cpp_cmd.binary_not_found_message()
                return
            _prewarm_error = None
            exit_code, output = await asyncio.to_thread(_run_pip_install)
            if exit_code != 0:
                _prewarm_error = output[-500:] if output else "pip install failed"
                return
            _prewarm_error = None

        if routing.peek_local_provider() is not provider:
            return

        if (
            _active_load is None
            or _active_load[0] is not provider
            or _active_load[1].done()
        ):
            # background-task-ok: strong ref held in _active_load; awaited via shield()
            _active_load = (provider, asyncio.create_task(_run_get_model(provider)))
        load_task = _active_load[1]

        await asyncio.shield(load_task)


async def await_local_ready(
    stt_settings: STTSettings, timeout: float | None = None
) -> bool:
    """Await the local STT provider's readiness before the request path uses it.
    Raises ``LocalReadinessTimeoutError`` after ``timeout`` seconds, or after
    ``_READY_TIMEOUT`` read at call time when it is ``None``; that is the one
    outcome a caller must treat as fatal. ``False`` means not loaded, not failed.
    """
    if timeout is None:
        timeout = _READY_TIMEOUT
    try:
        await asyncio.wait_for(ensure_local_ready(stt_settings), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise LocalReadinessTimeoutError(
            f"Local speech-to-text model did not become ready within {timeout:.0f}s"
        ) from e

    provider = routing.peek_local_provider()
    return provider is not None and provider.is_loaded


def _estimate_model_ram_mb() -> int | None:
    """Approximate the backend RSS-delta consumed by the loaded whisper model.

    The current process RSS in MB, or `None` for the whisper.cpp-server kind,
    whose model memory lives in a separate process's address space.
    """
    if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_SERVER:
        return None
    try:
        import os

        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss
        return rss // (1024 * 1024)
    except Exception:
        return None


def _check_package_installed() -> bool:
    """Check if the platform/kind-appropriate local STT dependency is present.

    The whisper.cpp kind resolves the `whisper-server` binary — nothing is
    pip-installable for it; everywhere else imports `faster_whisper`.
    """
    if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_SERVER:
        return local_whisper_cpp_cmd.resolve_binary_path() is not None

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

    if getattr(sys, "frozen", False):
        yield sse_event("error", {
            "status": "error",
            "error": (
                "Local STT install is not supported in the packaged build. "
                "Install JustSay from source if you need Local mode on this OS."
            ),
        })
        return

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
                yield sse_event(
                    "error",
                    {"status": "error", "error": output[-500:] if output else "pip install failed"},
                )
        except Exception as e:
            log.warning("pip install failed: %s", e)
            yield sse_event("error", {"status": "error", "error": str(e)})


def _run_pip_install() -> tuple[int, str]:
    """Run pip install .[local] synchronously. Returns (exit_code, output).

    One extras name for every platform with a pip path; the accelerated ones
    resolve a bundled binary and never reach here.
    """
    backend_dir = _get_backend_dir()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", ".[local]"],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout + result.stderr
    log.info("pip install exit code: %d", result.returncode)
    if result.returncode != 0:
        log.warning("pip install output: %s", output[-1000:])

    return result.returncode, output


def _get_backend_dir():
    """Get the backend project directory (where pyproject.toml lives)."""
    from pathlib import Path

    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    return Path(__file__).resolve().parent.parent.parent


def _detect_gpu() -> tuple[bool, str | None, str]:
    """Detect GPU availability, a human-readable device name, and vendor.

    Returns `(available, device_name_or_none, vendor)`; `available` is
    NVIDIA/CUDA-only and feeds the faster-whisper "auto" device decision alone.
    """
    if is_macos_arm64():
        return True, "Apple Silicon (Metal)", "apple"

    from app.core.gpu_probe import GpuVendor, probe_gpu

    result = probe_gpu()
    available = result.vendor == GpuVendor.NVIDIA
    return available, result.name, result.vendor.value
