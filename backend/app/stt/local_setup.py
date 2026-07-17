"""STT Local mode readiness checks — package detection, GPU, pip install."""

import asyncio
import logging
import subprocess
import sys
from collections.abc import AsyncIterator

from pydantic import BaseModel

from app.core.types import ProviderMode
from app.core.utils import sse_event
from app.stt.config import STTSettings
from app.stt.local_factory import LocalProviderKind, get_local_provider_kind, is_macos_arm64

log = logging.getLogger(__name__)

_install_lock = asyncio.Lock()

# Serializes the entire probe -> install-if-needed -> load attempt end to
# end, so a fast Local<->Cloud flap collapses to at most one real attempt
# instead of racing multiple overlapping loads. Must be an asyncio.Lock, not
# threading.Lock: the critical section spans genuine `await`s (pip install
# via asyncio.to_thread, the model load itself) — mirrors
# app.embeddings.resolve_embedding_provider's (LOCAL, LOCAL) probe/decide/
# cleanup-or-reuse lock (spec 013 precedent).
_prewarm_lock = asyncio.Lock()
# Surfaced through check_status().last_error alongside the provider's own
# last_load_error — set on an install failure (before a provider-level error
# could even occur), cleared on a successful install or a fresh install
# attempt.
_prewarm_error: str | None = None


class LocalSttStatus(BaseModel):
    package_installed: bool = False
    model_loaded: bool = False
    model_name: str = ""
    model_ram_mb: int | None = None
    gpu_available: bool = False
    gpu_name: str | None = None
    # "apple" on macOS arm64, else the app.core.gpu_probe vendor value
    # ("nvidia"/"amd"/"intel"/"none"). Populated even when gpu_available is
    # False (e.g. an explicit whisper_device="cpu" override) — AMD/Intel
    # Windows is Vulkan-accelerated (see WHISPER_CPP_VULKAN in local_factory.py).
    gpu_vendor: str = "none"
    device: str = "cpu"
    compute_type: str = "int8"
    last_error: str | None = None


def check_status(stt_settings: STTSettings) -> LocalSttStatus:
    """Check local STT readiness: package installed + load state + GPU + last error."""
    installed = _check_package_installed()
    # `cuda_probe_available` feeds only the "auto" -> cuda/cpu decision below
    # (faster-whisper/CTranslate2 has no AMD/Intel backend, so that decision
    # stays NVIDIA-only) — the status object's own `gpu_available` field is
    # computed separately, from the final resolved `device`, further down.
    cuda_probe_available, gpu_name, gpu_vendor = _detect_gpu()

    if is_macos_arm64():
        device = "mlx"
        # Informational label — actual dtype is controlled inside mlx-whisper.
        # Accurate for the project default large-v3-turbo and other large
        # variants; smaller MLX checkpoints ship as float16.
        compute_type = "bfloat16"
    elif get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_VULKAN:
        device = "vulkan"
        compute_type = "float16"
    else:
        device = stt_settings.whisper_device
        if device == "auto":
            device = "cuda" if cuda_probe_available else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

    # True whenever the *final resolved* device indicates real GPU
    # acceleration — not just NVIDIA — so a Vulkan-accelerated AMD/Intel
    # session never reports the contradictory device: "vulkan" +
    # gpu_available: false pair (Stage 3 review, iteration 1, issue #2).
    gpu_available = device in ("cuda", "vulkan", "mlx")

    # is_model_loaded reads the cached provider; safe even if the package is missing.
    from app.stt import get_local_load_error, is_model_loaded

    # Mutually exclusive in practice: an install failure returns before any
    # provider-level error could be set, and a load failure only ever
    # happens after install already succeeded (_prewarm_error is None by then).
    last_error = get_local_load_error(stt_settings) or _prewarm_error

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
        last_error=last_error,
    )


def maybe_prewarm_local(stt_settings: STTSettings) -> None:
    """Fire-and-forget. No-op unless ``stt_settings.mode`` is LOCAL.

    Called from every place the active STT mode can change or need
    re-warming: ``set_stt_mode()``, ``put_settings()``, and ``lifespan()`` —
    not just the literal toggle click, so Local mode is always warm by the
    time the first dictation request needs it.
    """
    if stt_settings.mode != ProviderMode.LOCAL:
        return
    asyncio.create_task(ensure_local_ready(stt_settings))


async def ensure_local_ready(stt_settings: STTSettings) -> None:
    """Install (if needed) and load the Local STT model, serialized through
    ``_prewarm_lock`` so overlapping calls collapse into one real attempt.

    The entry check is ``stt_settings.mode``-based (no point starting an
    attempt at all once mode has already moved on). The mid-install and
    mid-load rechecks are cache-*identity* checks instead
    (``peek_local_provider() is not provider``), not mode checks — a mode
    check is structurally insufficient here: ``clear_cache()`` can evict the
    captured ``provider`` from the cache without ``stt_settings.mode`` ever
    changing (e.g. an unrelated ``PUT /settings`` edit routed through
    ``sync_to_runtime()``'s ``changed_stt`` branch while Local stays active
    the whole time — spec 015, RED-1). The identity check is strictly more
    general: it still catches every genuine Local -> Cloud switch (which
    itself goes through ``clear_cache()``), plus the mode-stays-LOCAL case a
    mode check would miss entirely. If the cache moved on, the now-orphaned
    provider is cleaned up once the load settles (success or failure) —
    regardless of what ``stt_settings.mode`` currently says.
    """
    global _prewarm_error
    async with _prewarm_lock:
        if stt_settings.mode != ProviderMode.LOCAL:
            return  # superseded before this attempt even started

        from app.stt import get_provider, peek_local_provider

        provider = get_provider(ProviderMode.LOCAL, stt_settings)
        if provider.is_loaded:
            return  # already warm — the common case after the first prewarm

        if not _check_package_installed():
            if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_VULKAN:
                # Nothing to `pip install` for this kind — the whisper-server
                # binary is either bundled/dev-vendored or it isn't.
                _prewarm_error = (
                    "whisper-server binary not found. Set JUSTSAY_WHISPER_CPP_BIN, "
                    "or run backend/scripts/build_whisper_cpp_vulkan.ps1 for local dev."
                )
                return
            _prewarm_error = None
            exit_code, output = await asyncio.to_thread(_run_pip_install)
            if exit_code != 0:
                _prewarm_error = output[-500:] if output else "pip install failed"
                return
            _prewarm_error = None

        # Identity check, NOT a stt_settings.mode check — see docstring above.
        if peek_local_provider() is not provider:
            return  # cache moved on before we even started the model load

        try:
            await asyncio.to_thread(provider._get_model)
        except Exception:
            pass  # provider._last_load_error is already latched
        finally:
            if peek_local_provider() is not provider:
                provider.cleanup()  # orphaned — cache moved on mid-load


def _estimate_model_ram_mb() -> int | None:
    """Approximate the backend RSS-delta consumed by the loaded whisper model.

    Returns the current process RSS in MB — coarse but informative; the user
    sees "the backend is holding ~700 MB" rather than no number at all.

    Returns `None` for the Vulkan kind: the actual model memory lives in the
    separate `whisper-server` child process's own address space, not this
    (the FastAPI backend's) process's RSS — reporting the wrong process's
    RSS would be actively misleading rather than merely imprecise.
    """
    if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_VULKAN:
        return None
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
    """Check if the platform/kind-appropriate local STT dependency is present.

    macOS arm64 checks for the importable `mlx_whisper` package. Windows
    AMD/Intel checks whether the whisper.cpp `whisper-server` binary can be
    resolved (bundled resource dir, dev-vendor dir, or env override) —
    there's nothing to `pip install` for that kind, the binary is either
    bundled or it isn't. Everywhere else checks for the importable
    `faster_whisper` package.
    """
    if is_macos_arm64():
        try:
            import mlx_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    if get_local_provider_kind() == LocalProviderKind.WHISPER_CPP_VULKAN:
        from app.stt import local_vulkan_cmd

        return local_vulkan_cmd.resolve_binary_path() is not None

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
    Everywhere else, delegates to `app.core.gpu_probe.probe_gpu()`. The
    returned `available` bool stays NVIDIA/CUDA-only (faster-whisper/
    CTranslate2 has no AMD/Intel backend) — it feeds only `check_status()`'s
    "auto" -> cuda/cpu decision for that provider, not the status object's
    own `gpu_available` field, which is computed separately in
    `check_status()` from the final resolved `device` and also covers the
    Vulkan-accelerated AMD/Intel path. `gpu_name`/vendor are always
    populated when a GPU is detected, regardless of which provider ends up
    accelerated.

    Returns (available, device_name_or_none, vendor).
    """
    if is_macos_arm64():
        return True, "Apple Silicon (MLX/Metal)", "apple"

    from app.core.gpu_probe import GpuVendor, probe_gpu

    result = probe_gpu()
    available = result.vendor == GpuVendor.NVIDIA
    return available, result.name, result.vendor.value
