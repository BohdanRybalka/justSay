"""whisper.cpp + Vulkan local STT provider -- Windows AMD/Intel GPU acceleration.

Selected by `app.stt.local_factory.get_local_provider_class()` on Windows
when `app.core.gpu_probe.probe_gpu()` reports AMD or Intel -- faster-whisper
(CTranslate2, `LocalSTTProvider`) has no AMD backend at all, and
whisper.cpp's Vulkan backend runs on NVIDIA/AMD/Intel through the same
Vulkan API with no per-vendor code path. Full design rationale in
`docs/adr/011-whisper-cpp-vulkan-stt-provider.md`.

Runs whisper.cpp's `whisper-server` binary as a **persistent** local HTTP
child process (never spawned per-request -- a one-shot `whisper-cli`
invocation would reload a multi-GB GGML model into VRAM on every single
dictation, making the "accelerated" path slower than the CPU fallback it
replaces). `transcribe()` talks to the already-running server over
`http://127.0.0.1:<port>/inference`.
"""

import asyncio
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from app.stt.base import STTProvider, TranscriptionResult
from app.stt.config import STTSettings
from app.stt.local_vulkan_cmd import build_server_argv, resolve_binary_path, resolve_model_path

log = logging.getLogger(__name__)

_HOST = "127.0.0.1"
# Arbitrary, fixed local-only port -- distinct from the main backend's own
# hardcoded 9377 (src-tauri/src/backend.rs::PORT). Single-user local desktop
# app; a fixed constant carries the same low collision risk the main
# backend already accepts (see plan 018's Cuts deferred).
_PORT = 8878

_HEALTH_POLL_INTERVAL = 0.5
# ~120s -- generous for a first-run cold load of a multi-GB model into VRAM.
# whisper-server loads the model BEFORE binding its listen socket (confirmed
# by reading examples/server/server.cpp), so there is no incremental
# "still loading" signal -- a refused connection just means "not ready yet".
_HEALTH_POLL_MAX_ATTEMPTS = 240
_HEALTH_REQUEST_TIMEOUT = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=2.0)
_INFERENCE_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=120.0, pool=120.0)
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)

_GRACE_POLL_INTERVAL = 0.1
_GRACE_POLL_MAX_ATTEMPTS = 30  # 3s, mirrors backend.rs's terminate_gracefully() grace window

# Serializes "someone is tearing down a whisper-server on _PORT" against
# "someone is spawning a whisper-server on _PORT" -- module-level (not an
# instance attribute) because the port itself is a process-wide resource
# shared across provider *instances*: `clear_cache()` always replaces the
# cached provider with a brand-new instance on every Local<->Cloud switch
# (app/stt/__init__.py::_get_or_create), so the dying old instance and the
# spawning new instance share no instance state to synchronize on directly.
# Never held from the FastAPI event-loop thread: `_terminate_process()` only
# ever runs from `cleanup()`'s background daemon thread or from `_get_model`'s
# except-branch, and `_spawn_server()` only ever runs from `_get_model`,
# itself always invoked via `asyncio.to_thread` (transcribe(), router.py's
# `POST /stt/local/load`, local_setup.ensure_local_ready() -- confirmed by
# reading all three call sites). See plan 018, Review history iteration 2,
# RED-1.
_port_lock = threading.Lock()

# Serializes "someone is streaming the GGML model to model_path.part" against
# a second, independent WhisperCppVulkanSTTProvider instance doing the exact
# same thing -- module-level for the same reason _port_lock is: clear_cache()
# always replaces the cached provider with a brand-new instance on every
# Local<->Cloud switch, so an old instance's in-flight download (e.g. Spec
# 015's eager pre-warm) and a new instance's own _get_model() call share no
# instance state to synchronize on directly. A single lock covering the whole
# download step (not one keyed per model_size) is simplest and sufficient --
# single-user local desktop app, same rationale as _port_lock's own choice of
# a single module-level lock over per-resource locking. See GitHub review on
# PR #21, iteration 1, issue #1.
_download_lock = threading.Lock()

# ggerganov/whisper.cpp's own Hugging Face model repo -- GGML format,
# distinct from the CTranslate2 format faster-whisper auto-downloads for
# LocalSTTProvider (a real, permanent second-model-file cost, see plan 018's
# Risks).
_HF_MODEL_URL_TEMPLATE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class WhisperCppVulkanSTTProvider(STTProvider):
    """whisper.cpp + Vulkan -- Windows AMD/Intel local privacy-first STT provider.

    Model is auto-downloaded on first use (GGML format, ~1.6 GB for
    large-v3-turbo) into ``~/.justsay/models/whisper-cpp/``. The
    ``whisper-server`` binary is bundled with the app (Windows release) or
    resolved from a local dev-vendor directory -- see
    ``local_vulkan_cmd.resolve_binary_path()``.
    """

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._server_ready: bool = False
        self._last_load_error: str | None = None
        # Sync primitive -- `_get_model` runs both directly (from
        # `transcribe()` via `asyncio.to_thread`) and from router.py's
        # `POST /stt/local/load` (also via `asyncio.to_thread`) -- same
        # genuine OS-thread-race rationale as `LocalSTTProvider._load_lock`.
        self._load_lock: threading.Lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"whisper-cpp-vulkan/{self._settings.whisper_model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._server_ready

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _get_model(self) -> None:
        """Sync lazy-load entrypoint: resolve the binary, lazy-download the
        GGML model if missing, spawn `whisper-server` once, health-poll it,
        then mark `is_loaded=True`.

        Named `_get_model` -- not `_ensure_server`, this method's conceptual
        name in the ADR -- because `router.py`'s `POST /stt/local/load` and
        `local_setup.ensure_local_ready()` both call `provider._get_model`
        unconditionally on whatever provider `get_provider(LOCAL, ...)`
        returns. The sibling providers' `_get_model` name is a load-bearing
        duck-typed convention across concrete `STTProvider`s, not a
        documented part of the ABC itself -- confirmed by reading
        `app/stt/router.py` and `app/stt/local_setup.py` directly.
        """
        with self._load_lock:
            if self._server_ready and self._process is not None and self._process.poll() is None:
                return  # already warm -- the common case after the first load
            try:
                binary_path = resolve_binary_path()
                if binary_path is None:
                    raise RuntimeError(
                        "whisper-server binary not found. Set JUSTSAY_WHISPER_CPP_BIN, "
                        "or run backend/scripts/build_whisper_cpp_vulkan.ps1 for local dev."
                    )
                model_path = resolve_model_path(self._settings.whisper_model_size)
                if not model_path.is_file():
                    self._download_model(model_path)

                # Held only around the spawn itself -- not the download or
                # health-poll steps, neither of which touches the port --
                # so a new spawn blocks until any in-flight termination of a
                # previous instance's process (see _terminate_process()) has
                # genuinely finished, guaranteeing _PORT is actually free
                # before the Popen() call.
                with _port_lock:
                    self._spawn_server(binary_path, model_path)
                self._wait_until_healthy()
                self._server_ready = True
                self._last_load_error = None
                log.info(
                    "whisper-server ready: model=%s port=%d",
                    self._settings.whisper_model_size, _PORT,
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._last_load_error = msg
                self._server_ready = False
                log.exception("whisper-server load failed: %s", msg)
                # _spawn_server() may have succeeded even though a later step
                # (model download happens earlier, but _wait_until_healthy()
                # can still time out) raised -- terminate the orphaned child
                # and clear the handle so a subsequent _get_model() call spawns
                # a genuinely fresh process instead of silently overwriting a
                # still-alive one and leaking it (VRAM + a port-8878-holding
                # zombie).
                self._terminate_process(self._process)
                self._process = None
                raise

    def _download_model(self, model_path: Path) -> None:
        """Stream the GGML model to a `.part` temp file, renaming only on a
        fully-successful download -- a partial download from an interrupted
        first run must never be mistaken for a complete model on the next
        launch.

        Holds `_download_lock` for the entire body: `_get_model()`'s own
        `model_path.is_file()` check happens *before* this method is called,
        so two independent provider instances (e.g. an old instance's Spec
        015 eager pre-warm still downloading, racing a new instance created
        by a rapid Local->Cloud->Local switch) can both decide the model is
        missing and both call in here. Re-checks `model_path.is_file()` right
        after acquiring the lock so whichever instance loses the race skips
        the redundant multi-GB re-download entirely once it sees the winner
        already finished, instead of interleaving writes into the same
        `.part` file.
        """
        with _download_lock:
            if model_path.is_file():
                log.info(
                    "GGML model already downloaded by a concurrent instance -- "
                    "skipping redundant download: %s", model_path,
                )
                return

            url = _HF_MODEL_URL_TEMPLATE.format(size=self._settings.whisper_model_size)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            part_path = model_path.with_name(model_path.name + ".part")

            log.info("Downloading GGML model: %s -> %s", url, model_path)
            with httpx.Client(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(part_path, "wb") as f:
                        for chunk in resp.iter_bytes(_DOWNLOAD_CHUNK_SIZE):
                            f.write(chunk)
            part_path.replace(model_path)
            log.info("GGML model download complete: %s", model_path)

    def _spawn_server(self, binary_path: Path, model_path: Path) -> None:
        argv = build_server_argv(binary_path, model_path, _HOST, _PORT)
        creationflags = 0
        if sys.platform == "win32":
            # Same 0x08000000 value src-tauri/src/backend.rs already uses
            # for its own child spawns -- same suppress-console-flash
            # intent, one process layer down.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log.info("Spawning whisper-server: %s", argv)
        self._process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def _terminate_process(self, process: subprocess.Popen | None) -> None:
        """`.terminate()` -> grace-poll -> `.kill()` fallback for one process.

        Synchronous and blocking (up to `_GRACE_POLL_MAX_ATTEMPTS *
        _GRACE_POLL_INTERVAL` plus a further blocking `wait(timeout=3.0)` on
        the kill fallback) -- callers that must not block the FastAPI
        event-loop thread (`cleanup()`) run this on a background daemon
        thread instead of calling it directly.

        Holds `_port_lock` for its entire body -- this is the one helper
        both `cleanup()`'s background thread and `_get_model()`'s
        except-branch orphan-cleanup already funnel through, so serializing
        here serializes both callers against `_get_model()`'s own
        `_port_lock`-guarded spawn without either call site needing its own
        locking.

        Never itself raises: the `.kill()` fallback's `wait(timeout=3.0)`
        can raise `subprocess.TimeoutExpired` if the process is still
        stubbornly alive after being killed (e.g. a hung Vulkan driver or AV
        interference) -- that's logged, not propagated, so callers (notably
        `_get_model()`'s except-branch) are guaranteed to reach their own
        cleanup (`self._process = None`) and callers see only the original
        triggering error, not a confusing `TimeoutExpired` traceback.
        """
        if process is None:
            return
        with _port_lock:
            if process.poll() is not None:
                return
            log.info("Terminating whisper-server (pid=%s)", process.pid)
            process.terminate()
            for _ in range(_GRACE_POLL_MAX_ATTEMPTS):
                if process.poll() is not None:
                    break
                time.sleep(_GRACE_POLL_INTERVAL)
            else:
                log.warning("whisper-server still alive after grace period -- killing")
                process.kill()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    log.error(
                        "whisper-server (pid=%s) did not exit within 3s of being killed",
                        process.pid,
                    )

    def _wait_until_healthy(self) -> None:
        """Poll `GET /health` until it reports 200 `{"status":"ok"}`, the
        child exits early, or the attempt budget is exhausted.
        """
        assert self._process is not None
        url = f"http://{_HOST}:{_PORT}/health"
        with httpx.Client(timeout=_HEALTH_REQUEST_TIMEOUT) as client:
            for _ in range(_HEALTH_POLL_MAX_ATTEMPTS):
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"whisper-server exited early (code {self._process.returncode})"
                    )
                try:
                    r = client.get(url)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(_HEALTH_POLL_INTERVAL)
        raise RuntimeError("whisper-server did not become healthy within the poll budget")

    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe via the already-running whisper-server's `POST /inference`.

        `style`/`audio_duration` kwargs are accepted for interface parity
        but ignored -- this spec's own scope explicitly does not replicate
        `LocalSTTProvider`'s duration-driven beam_size/VAD tuning (plan 018,
        Cuts deferred: proving the base accelerated path works is the job;
        quality/latency tuning is a follow-up).
        """
        await asyncio.to_thread(self._get_model)

        url = f"http://{_HOST}:{_PORT}/inference"
        data = {"language": language, "response_format": "json"}

        log.info(
            "whisper-server: transcribe model=%s file=%s lang=%s",
            self._settings.whisper_model_size, audio_path.name, language,
        )

        def _post() -> str:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/wav")}
                with httpx.Client(timeout=_INFERENCE_TIMEOUT) as client:
                    resp = client.post(url, data=data, files=files)
            resp.raise_for_status()
            body = resp.json()
            raw_text = body.get("text", "")
            # whisper-server's `output_str()` joins segments with "\n" (no
            # embedded timestamps in the json/verbose_json `text` field) --
            # collapse to a single space-joined line, matching
            # LocalSTTProvider's output shape.
            return " ".join(line.strip() for line in raw_text.splitlines() if line.strip())

        text = await asyncio.to_thread(_post)
        return TranscriptionResult(text=text, tokens_used=None)

    def cleanup(self) -> None:
        """Terminate the whisper-server child.

        Mirrors `LocalSTTProvider.cleanup()`/`MLXWhisperSTTProvider.cleanup()`'s
        established non-blocking-lock-guard shape exactly: `cleanup()` is
        reachable synchronously from `PUT /stt/mode`'s `clear_cache()` on the
        FastAPI event-loop thread, so it must never block on `_load_lock`
        for the duration of a multi-minute first-run download/spawn. If the
        lock is busy, log and return without touching the process handle --
        the load's own caller is responsible for cleaning up an orphaned
        load afterwards (e.g. `ensure_local_ready()`'s post-load identity
        recheck).

        The lock-acquire/bookkeeping above is cheap and stays synchronous,
        but the actual `.terminate()` -> grace-poll -> `.kill()` sequence
        (up to ~6s: `_GRACE_POLL_MAX_ATTEMPTS * _GRACE_POLL_INTERVAL` plus a
        further blocking `wait(timeout=3.0)` on the kill fallback) runs on a
        background daemon thread instead, so `cleanup()` itself returns
        immediately -- matching the sibling providers' near-instant
        `cleanup()` contract that `PUT /stt/mode` already relies on (an
        `async def` endpoint calling `clear_cache()` synchronously,
        unawaited).
        """
        if not self._load_lock.acquire(blocking=False):
            log.info("cleanup() skipped: a server load is in flight (lock busy)")
            return
        try:
            self._server_ready = False
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                threading.Thread(
                    target=self._terminate_process, args=(process,), daemon=True
                ).start()
        finally:
            self._load_lock.release()
