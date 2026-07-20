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

# (provider, Task) for whichever _get_model() attempt is currently in
# flight, if any -- tracked independently of _prewarm_lock (Stage 5 GitHub
# review on PR #34, finding 1). `asyncio.wait_for`'s timeout cancels the
# *waiting coroutine*, not the worker thread `_get_model()` actually runs
# on -- a thread cannot be cancelled. Without this, a caller whose own wait
# times out releases `_prewarm_lock` while its `_get_model()` call keeps
# running unobserved; the NEXT caller then sees `is_loaded=False` (the
# orphaned load hasn't finished), passes every check, and starts a
# genuinely SECOND `_get_model()` call. Storing the in-flight (provider,
# Task) here lets a later caller find and re-join the SAME attempt via
# `asyncio.shield()` in `ensure_local_ready()` instead of starting a
# redundant one -- `asyncio.shield()` is what lets a caller's own timeout
# detach it from *observing* the task without cancelling the task itself.
_active_load: tuple[object, asyncio.Task] | None = None

# Generous enough to cover a genuinely cold first run -- pip install (up to
# 300s per _run_pip_install's own subprocess timeout) plus the model load
# itself -- so this bound only ever trips on a load that is truly stuck
# (broken GPU driver, dead network mid-download), not on an ordinary slow
# first-time setup. See await_local_ready().
_READY_TIMEOUT = 300.0


def peek_active_load() -> asyncio.Task | None:
    """Read-only view of the in-flight model-load task, if any.

    Exists so `lifespan()`'s shutdown drain can cancel it. `_active_load` is
    deliberately NOT registered in `app.core.tasks` (it holds its own strong
    reference and its exception is retrieved via `shield()`), so the drain
    cannot reach it through the registry -- but leaving it running while
    `clear_stt()` fires is exactly the race this accessor closes.
    """
    return _active_load[1] if _active_load is not None else None


class LocalReadinessTimeout(Exception):
    """Raised by await_local_ready() when the bounded wait genuinely times
    out -- i.e. ensure_local_ready() itself did not return within the
    budget, as opposed to returning promptly via one of its own early-return
    guards (see await_local_ready()'s docstring for why that distinction
    matters)."""


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
    else:
        # Pass the vendor _detect_gpu() already resolved straight through to
        # get_local_provider_kind() instead of letting it call probe_gpu() a
        # second time — probe_gpu() has no caching (docs/TODO.md → Tech
        # Debt), and check_status() is polled every 3s by the Settings tab.
        # gpu_vendor is always "nvidia"/"amd"/"intel"/"none" here (the
        # "apple" value only comes back from the is_macos_arm64() branch of
        # _detect_gpu(), which can't be true in this else branch since it's
        # the same is_macos_arm64() check).
        from app.core.gpu_probe import GpuVendor

        kind = get_local_provider_kind(GpuVendor(gpu_vendor))
        if kind == LocalProviderKind.WHISPER_CPP_VULKAN:
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
    re-warming: ``set_stt_mode()``, ``put_settings()``, and the manual
    ``POST /stt/local/prewarm`` retry — not just the literal toggle click, so
    Local mode is always warm by the time the first dictation request needs
    it. NOT called from ``lifespan()`` — the automatic every-process-start
    trigger goes through ``maybe_prewarm_local_at_startup()`` instead, which
    adds crash-loop protection this function deliberately does not have (see
    that function's docstring).

    [Spec 023] Resets the startup crash-loop guard's on-disk counter to 0
    on every explicit trigger (mode switch, an STT-relevant settings edit,
    or the manual POST /stt/local/prewarm retry) -- a deliberate,
    user-initiated attempt is a fresh start, decoupled from the automatic
    every-restart streak maybe_prewarm_local_at_startup() tracks.
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
    """Startup-only entry point, called once from app.main.lifespan().

    Guards against a crash loop: if the model load itself crashes the
    process, an unconditional prewarm on every Spec 011 watchdog respawn
    would re-attempt the same doomed load. Not used by set_stt_mode()/
    put_settings()/the manual POST /stt/local/prewarm retry -- all three
    call maybe_prewarm_local() directly and require an already-running,
    already-healthy backend to receive the request, so none of them is part
    of the automatic every-process-start loop this guards against.
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
    tasks.spawn_background_task(
        _prewarm_then_clear_crash_guard(stt_settings), name="local-stt-prewarm-startup"
    )


async def _prewarm_then_clear_crash_guard(stt_settings: STTSettings) -> None:
    try:
        await ensure_local_ready(stt_settings)
    finally:
        _write_consecutive_incomplete_prewarms(0)


async def _run_get_model(provider) -> None:
    """The actual ``_get_model()`` attempt, run as an independent
    ``asyncio.Task`` (created in ``ensure_local_ready`` below) rather than
    awaited directly inline. That is what makes ``_active_load`` meaningful:
    a Task keeps running to completion even if every caller currently
    watching it (via ``asyncio.shield()``) gets cancelled -- unlike a plain
    coroutine awaited in place, which a cancellation unwinds immediately.
    Same swallow-and-latch / orphan-cleanup contract ``ensure_local_ready``
    always had.
    """
    try:
        await asyncio.to_thread(provider._get_model)
    except Exception:
        pass  # provider._last_load_error is already latched
    finally:
        from app.stt import peek_local_provider

        if peek_local_provider() is not provider:
            provider.cleanup()  # orphaned — cache moved on mid-load


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

    The actual ``_get_model()`` call runs as an ``asyncio.Task``
    (``_active_load``), awaited here via ``asyncio.shield()`` rather than
    directly (Stage 5 GitHub review on PR #34, finding 1). If THIS caller
    is itself cancelled (e.g. by ``await_local_ready()``'s own
    ``wait_for`` timing out), ``shield()`` detaches only this caller's
    *observation* of the task -- the task, and the worker thread
    ``_get_model()`` runs on (which cannot be cancelled once started),
    keep going. A later caller that reaches this same function while that
    task is still running finds it in ``_active_load`` and re-joins it
    instead of starting a genuinely second ``_get_model()`` call.

    That ``asyncio.create_task()`` call (unlike every other fire-and-forget
    call site in this module, which route through
    ``app.core.tasks.spawn_background_task()``, Spec 032) is deliberately
    left bare: ``_active_load`` already holds its own strong reference for
    the task's whole lifetime, and its exception is already retrieved via
    the ``asyncio.shield()`` below, so wrapping it in the shared helper
    would add a redundant registry entry and a redundant exception
    retrieval for no correctness gain.

    ``asyncio.shield()`` is called from *inside* the ``_prewarm_lock``
    block, matching the lock's original scope, deliberately: moving it
    outside would let a second caller's own ``get_provider()`` lookup run
    concurrently with the first attempt's in-flight ``_get_model()`` side
    effects (e.g. a settings change clearing the provider cache mid-load),
    which changes the ordering spec 015's RED-1 orphan-cleanup regression
    test depends on. Keeping the lock's scope unchanged means the only
    behavioural difference from before is exactly the one this fix targets:
    what survives a caller's own cancellation.
    """
    global _prewarm_error, _active_load
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

    Reuses ensure_local_ready()'s own ``_prewarm_lock``, so a request arriving
    while a prewarm is already in flight blocks on that lock and returns once
    the *existing* load finishes -- no second ``_get_model()`` call (AC 11).

    Bounded by ``asyncio.wait_for(..., timeout=timeout)`` so a genuinely stuck
    load (dead network mid-download, a broken driver) cannot hang the request
    path indefinitely. On a real timeout this raises ``LocalReadinessTimeout``
    -- the one outcome callers SHOULD treat as fatal, since letting the
    request proceed risks an even longer, unbounded hang inside
    ``transcribe()``'s own lazy ``_get_model()`` fallback.

    Returns whether the active local provider ended up loaded. A ``False``
    return (no timeout, but not loaded either) covers ensure_local_ready()'s
    own fast early-return guards racing in -- ``stt_settings.mode`` flipping
    away from LOCAL, or the cache moving on to a different provider instance
    -- while this call was queued on ``_prewarm_lock``. Those are NOT
    failures: the caller must not treat a plain ``False`` as fatal, only a
    raised ``LocalReadinessTimeout``. `LocalSTTProvider.transcribe()` (and its
    Vulkan/MLX siblings) retain their own lazy ``_get_model()`` fallback, so a
    plain ``False`` return here is never fatal to the caller.

    This is a trade, not a guarantee that nothing changes (Stage 5 GitHub
    review on PR #34, finding 2 -- a prior version of this docstring claimed
    "never turn a working request into a failing one", which does not hold
    and should not have been written that way). A load that finishes within
    ``timeout`` behaves exactly as it did before this barrier existed. A
    load that would have EVENTUALLY succeeded but takes LONGER than
    ``timeout`` -- a genuinely slow but working cold start: a large model on
    a slow disk, a throttled first-time download -- is deliberately
    converted into an explicit ``LocalReadinessTimeout`` (surfaced by
    ``process_audio`` as a clear error) rather than the unbounded hang it
    used to be. That trade is intentional -- an indefinite hang is worse
    than a clear, actionable error -- but it does mean a request that would
    previously have succeeded, given enough time, can now fail instead.
    Choose ``timeout`` (or override it per call) with that trade in mind.

    ``timeout=None`` (the default) reads the module-level ``_READY_TIMEOUT``
    at call time rather than binding it as a default-argument value at
    function-definition time -- the latter would freeze the value at import
    time, defeating tests (and any future runtime override) that patch
    ``_READY_TIMEOUT`` directly, mirroring this module's own
    ``_HEALTH_POLL_MAX_ATTEMPTS``-style convention in ``local_vulkan.py``.
    """
    if timeout is None:
        timeout = _READY_TIMEOUT
    try:
        await asyncio.wait_for(ensure_local_ready(stt_settings), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise LocalReadinessTimeout(
            f"Local speech-to-text model did not become ready within {timeout:.0f}s"
        ) from e

    from app.stt import peek_local_provider

    provider = peek_local_provider()
    return provider is not None and provider.is_loaded


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
