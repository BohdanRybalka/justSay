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
import atexit
import ctypes
import logging
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import httpx

from app.stt.base import (
    STTProvider,
    TranscriptionResult,
    min_no_speech_prob,
    normalize_detected_language,
)
from app.stt.config import STTSettings
from app.stt.local_vulkan_cmd import build_server_argv, resolve_binary_path, resolve_model_path

log = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_PORT = 8878

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_MAX_ATTEMPTS = 240
_HEALTH_REQUEST_TIMEOUT = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=2.0)
_INFERENCE_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=120.0, pool=120.0)
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)

_GRACE_POLL_INTERVAL = 0.1
_GRACE_POLL_MAX_ATTEMPTS = 30

_port_lock = threading.Lock()

_download_lock = threading.Lock()

_HF_MODEL_URL_TEMPLATE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024



_live_children_lock = threading.Lock()
_live_children: dict[int, subprocess.Popen] = {}


def _register_child(process: subprocess.Popen) -> None:
    with _live_children_lock:
        _live_children[process.pid] = process


def _deregister_child(process: subprocess.Popen) -> None:
    with _live_children_lock:
        _live_children.pop(process.pid, None)


def _reap_orphans() -> None:
    """atexit hook: terminate any whisper-server child still registered when
    the interpreter exits -- i.e. one `_terminate_process()` (called from
    either `cleanup()` or `_get_model()`'s own except-branch orphan cleanup)
    never ran for it. A single `.terminate()` is deliberately simpler than
    `_terminate_process()`'s full terminate -> grace-poll -> kill sequence:
    this is the portable floor, not the real guarantee (the Windows Job
    Object is) -- see the module-level comment above.
    """
    with _live_children_lock:
        orphans = list(_live_children.values())
        _live_children.clear()
    for process in orphans:
        try:
            if process.poll() is None:
                log.warning(
                    "Reaping orphaned whisper-server (pid=%s) at interpreter "
                    "exit -- cleanup() was never called for it.", process.pid,
                )
                process.terminate()
        except Exception:
            log.exception(
                "Failed to reap orphaned whisper-server (pid=%s) at exit",
                getattr(process, "pid", "?"),
            )


atexit.register(_reap_orphans)



_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_job_object_lock = threading.Lock()
_job_object_handle: int | None = None
_job_object_init_failed = False
_kernel32_dll = None


def _kernel32():
    """Lazily load `kernel32` and declare explicit `restype`/`argtypes` on
    every Job Object call before first use (spec 028 iteration-2 review, AC
    16a). Without them, ctypes marshals return/argument values as 32-bit
    `c_int` by default, which silently truncates a real 64-bit `HANDLE` --
    it happens to work today only because handle values for a young process
    are small (observed 368/372 in review), which is luck, not a contract.
    """
    global _kernel32_dll
    if _kernel32_dll is None:
        dll = ctypes.WinDLL("kernel32", use_last_error=True)

        dll.CreateJobObjectW.restype = wintypes.HANDLE
        dll.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]

        dll.SetInformationJobObject.restype = wintypes.BOOL
        dll.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]

        dll.AssignProcessToJobObject.restype = wintypes.BOOL
        dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        _kernel32_dll = dll
    return _kernel32_dll


def _get_or_create_job_object() -> int | None:
    """Lazily create (once) a Windows Job Object configured with
    KILL_ON_JOB_CLOSE. Returns None (and never raises) if creation fails --
    callers degrade to the atexit registry as their only protection, which
    is layer 1's whole purpose."""
    global _job_object_handle, _job_object_init_failed
    with _job_object_lock:
        if _job_object_handle is not None or _job_object_init_failed:
            return _job_object_handle
        try:
            kernel32 = _kernel32()
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())

            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = kernel32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            _job_object_handle = handle
        except Exception:
            log.warning(
                "Failed to create Windows Job Object for whisper-server "
                "crash-safety -- falling back to atexit-only orphan reaping.",
                exc_info=True,
            )
            _job_object_init_failed = True
            return None
    return _job_object_handle


def _assign_to_job_object(process: subprocess.Popen) -> None:
    """Assign `process` to the shared Job Object so an ungraceful death of
    THIS Python process (including TerminateProcess, which runs no atexit
    handler) takes the child down with it. Windows-only; a no-op elsewhere.
    Never raises -- a failure here must degrade to the atexit registry, not
    break STT."""
    if sys.platform != "win32":
        return
    job = _get_or_create_job_object()
    if job is None:
        return
    try:
        kernel32 = _kernel32()
        proc_handle = int(process._handle)  # type: ignore[attr-defined]
        ok = kernel32.AssignProcessToJobObject(job, proc_handle)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        log.warning(
            "Failed to assign whisper-server (pid=%s) to the Windows Job "
            "Object -- falling back to atexit-only orphan reaping.",
            getattr(process, "pid", "?"), exc_info=True,
        )


class WhisperCppVulkanSTTProvider(STTProvider):
    """whisper.cpp + Vulkan -- Windows AMD/Intel local privacy-first STT provider.

    Model is auto-downloaded on first use (GGML format, ~1.6 GB for
    large-v3-turbo) into ``~/.justsay/models/whisper-cpp/``. The
    ``whisper-server`` binary is bundled with the app (Windows release) or
    resolved from a local dev-vendor directory -- see
    ``local_vulkan_cmd.resolve_binary_path()``.
    """

    is_local = True

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._server_ready: bool = False
        self._last_load_error: str | None = None
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
                return
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
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log.info("Spawning whisper-server: %s", argv)
        self._process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        _register_child(self._process)
        _assign_to_job_object(self._process)

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
        _deregister_child(process)
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

        `response_format` escalates to ``verbose_json`` only when
        ``language == "auto"`` -- plain ``"json"`` has no ``language`` field
        at all, but the explicit-language hot path (latency-sensitive, and
        on this Vulkan backend hardware this project cannot test against)
        keeps its exact current wire format unchanged (spec 029 / docs/adr/
        016-detected-language-on-stt-contract.md).
        """
        await asyncio.to_thread(self._get_model)

        url = f"http://{_HOST}:{_PORT}/inference"
        response_format = "verbose_json" if language == "auto" else "json"
        data = {"language": language, "response_format": response_format}

        log.info(
            "whisper-server: transcribe model=%s file=%s lang=%s format=%s",
            self._settings.whisper_model_size, audio_path.name, language, response_format,
        )

        def _post() -> tuple[str, str | None, float | None]:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/wav")}
                with httpx.Client(timeout=_INFERENCE_TIMEOUT) as client:
                    resp = client.post(url, data=data, files=files)
            resp.raise_for_status()
            body = resp.json()
            raw_text = body.get("text", "")
            text = "".join(raw_text.splitlines()).strip()
            no_speech_prob = (
                min_no_speech_prob(body.get("segments"))
                if response_format == "verbose_json"
                else None
            )
            return text, body.get("language"), no_speech_prob

        text, detected_raw, no_speech_prob = await asyncio.to_thread(_post)
        return TranscriptionResult(
            text=text,
            tokens_used=None,
            detected_language=normalize_detected_language(detected_raw),
            no_speech_prob=no_speech_prob,
        )

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
