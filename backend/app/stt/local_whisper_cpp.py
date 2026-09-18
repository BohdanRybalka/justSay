"""whisper.cpp `whisper-server` local STT provider -- one class, two GPU backends.

`app.stt.local_factory.get_local_provider_class` selects it on Windows with an
AMD or Intel GPU (Vulkan binary) and on macOS Apple Silicon (Metal binary); the
backend is baked into the binary, so both platforms share every line below
(ADR 011, ADR 036). That binary runs as a persistent local HTTP child process,
never one per request, and `transcribe()` posts to `/inference` on it.
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

from app.core.errors import ResourceUnavailableError
from app.stt.base import (
    STTProvider,
    TranscriptionResult,
    latched_load_error,
    min_no_speech_prob,
    normalize_detected_language,
)
from app.stt.config import STTSettings
from app.stt.local_whisper_cpp_cmd import (
    binary_not_found_message,
    build_server_argv,
    resolve_binary_path,
    resolve_model_path,
    vendor_dir_name,
)

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
    """atexit hook: terminate any whisper-server child still registered at exit.

    A single `.terminate()`, not `_terminate_process()`'s full sequence: this
    is the portable floor, and the Windows Job Object is the real guarantee.
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


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801
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


class _IO_COUNTERS(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801
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
    """Lazily load `kernel32` with explicit `restype`/`argtypes` on every Job
    Object call. Without them ctypes marshals as 32-bit `c_int` by default,
    which silently truncates a real 64-bit `HANDLE`.
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


class WhisperCppServerSTTProvider(STTProvider):
    """whisper.cpp `whisper-server` -- local privacy-first STT on Windows
    AMD/Intel (Vulkan) and macOS Apple Silicon (Metal). The GGML model
    auto-downloads on first use into ``~/.justsay/models/whisper-cpp/``.
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
        """``<vendor dir>/<model size>``, from the platform's vendor directory.

        Persisted into every history row, so it is a stored label rather than a
        display name; renaming it splits existing history across two labels.
        """
        return f"{vendor_dir_name() or 'whisper-cpp'}/{self._settings.whisper_model_size}"

    @property
    def is_loaded(self) -> bool:
        return self._server_ready

    @property
    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _get_model(self) -> None:
        """Sync lazy-load entrypoint: resolve the binary, download the GGML model
        if missing, spawn `whisper-server` once, health-poll it, then set
        `is_loaded`. The name is the duck-typed one every local provider owes.
        """
        with self._load_lock:
            if self._server_ready and self._process is not None and self._process.poll() is None:
                return
            try:
                binary_path = resolve_binary_path()
                if binary_path is None:
                    raise ResourceUnavailableError(binary_not_found_message())
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
                msg = latched_load_error(e)
                self._last_load_error = msg
                self._server_ready = False
                log.exception("whisper-server load failed: %s", msg)
                self._terminate_process(self._process)
                self._process = None
                raise

    def _download_model(self, model_path: Path) -> None:
        """Stream the GGML model to a `.part` file, renaming only on success.

        Holds `_download_lock` throughout and re-checks the model file after
        acquiring it, so a racing second instance skips the download entirely.
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

        Blocks for several seconds and holds `_port_lock` throughout, so a
        caller that must not block the event loop runs it on a thread. Never raises.
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
                    raise ResourceUnavailableError(
                        f"whisper-server exited early (code {self._process.returncode})"
                    )
                try:
                    r = client.get(url)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(_HEALTH_POLL_INTERVAL)
        raise ResourceUnavailableError(
            "whisper-server did not become healthy within the poll budget"
        )

    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe via the already-running whisper-server's `POST /inference`.

        `audio_duration` is accepted for parity and ignored; `response_format`
        escalates to ``verbose_json`` only when ``language == "auto"`` (ADR 016).
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

        Returns immediately: a load in flight wins `_load_lock` and this leaves
        the process handle alone, and the terminate sequence runs on a thread.
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
