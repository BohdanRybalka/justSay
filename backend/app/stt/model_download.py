"""The local speech model's install: one shared state, a resumable download, one transfer at a time.

The first load of the engine and the Install button both come through here, so a download
started by either shows the same progress in `snapshot()`. Install goes on past the download
into two more steps, checking the computer and starting the engine, which report their stage
here too.
"""

import errno
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from app.core.errors import ResourceUnavailableError

log = logging.getLogger(__name__)

HF_MODEL_URL_TEMPLATE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin"
_CHUNK_SIZE = 1024 * 1024
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)

_INTERRUPTED = "The download was interrupted. Check your internet connection and try again."
_INCOMPLETE = "The download ended before the whole model arrived. Try again."
_DISK_FULL = "There is not enough free space on the disk for the model."
_DISK_FAILED = "The model could not be saved to the disk."
_UNEXPECTED = "The download failed. See the JustSay log for the reason."


Step = Literal["downloading", "checking", "starting"]


class ModelDownloadState(BaseModel):
    """``bytes_total`` is ``None`` until the server has announced the size.

    ``failed_step`` names the step a ``failed`` install stopped in.
    """

    stage: Literal["idle", "downloading", "checking", "starting", "done", "failed"] = "idle"
    bytes_done: int = 0
    bytes_total: int | None = None
    error: str | None = None
    failed_step: Step | None = None


_ENGINE_STEPS = ("checking", "starting")

_download_lock = threading.Lock()
_state_lock = threading.Lock()
_state = ModelDownloadState()


def snapshot() -> ModelDownloadState:
    with _state_lock:
        return _state.model_copy()


def _update(**fields) -> None:
    global _state
    with _state_lock:
        _state = _state.model_copy(update=fields)


def _begin() -> None:
    _update(stage="downloading", bytes_done=0, bytes_total=None, error=None, failed_step=None)


def advance(stage: Literal["checking", "starting", "done"]) -> None:
    """Move the shared state to ``stage``, clearing an earlier failure."""
    _update(stage=stage, error=None, failed_step=None)


def fail(step: Step, sentence: str) -> None:
    """Record that the install stopped in ``step``, with a plain ``sentence`` for the user."""
    log.warning("Local model install failed in the %s step: %s", step, sentence, exc_info=True)
    _update(stage="failed", error=sentence, failed_step=step)


def download(model_size: str, model_path: Path) -> None:
    """Fetch the ``model_size`` model to ``model_path`` and return once it is there.

    A second caller waits for the first, then finds the file or transfers it again. Raises
    ``ResourceUnavailableError`` with a plain sentence and leaves a partial file for a resume.
    """
    with _download_lock:
        if model_path.is_file():
            return
        _begin()
        _transfer(model_size, model_path, finished="done")


def start(model_size: str, model_path: Path, after: Callable[[], None]) -> ModelDownloadState:
    """Install on a background thread: download the model if it is missing, then run ``after``.

    ``after`` owns the stages from ``checking`` on: it ends them in ``done`` or ``fail`` and
    never raises. A no-op while an install is past its download or still in it.
    """
    if not _download_lock.acquire(blocking=False):
        return snapshot()
    if snapshot().stage in _ENGINE_STEPS:
        _download_lock.release()
        return snapshot()
    if model_path.is_file():
        advance("checking")
    else:
        _begin()
    worker = threading.Thread(
        target=_install,
        args=(model_size, model_path, after),
        name="model-install",
        daemon=True,
    )
    try:
        worker.start()
    except BaseException:
        _update(stage="idle")
        _download_lock.release()
        raise
    return snapshot()


def _install(model_size: str, model_path: Path, after: Callable[[], None]) -> None:
    try:
        if not model_path.is_file():
            _transfer(model_size, model_path, finished="checking")
    except ResourceUnavailableError:
        return
    finally:
        _download_lock.release()
    after()


def _transfer(model_size: str, model_path: Path, finished: Literal["checking", "done"]) -> None:
    url = HF_MODEL_URL_TEMPLATE.format(size=model_size)
    part_path = model_path.with_name(model_path.name + ".part")
    log.info("Downloading GGML model: %s -> %s", url, model_path)
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
            _stream_into(client, url, part_path)
        part_path.replace(model_path)
    except ResourceUnavailableError as exc:
        fail("downloading", str(exc))
        raise
    except Exception as exc:
        sentence = _failure_sentence(exc)
        fail("downloading", sentence)
        raise ResourceUnavailableError(sentence) from exc
    advance(finished)
    log.info("GGML model download complete: %s", model_path)


def _stream_into(client: httpx.Client, url: str, part_path: Path) -> None:
    """Append what the server sends to ``part_path``, resuming where a partial file ends."""
    resume_from = part_path.stat().st_size if part_path.is_file() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 416:
            if _size_after_slash(response.headers.get("Content-Range")) == resume_from:
                _update(bytes_done=resume_from, bytes_total=resume_from)
                return
            part_path.unlink()
            _stream_into(client, url, part_path)
            return
        response.raise_for_status()
        resumed = response.status_code == 206
        offset = resume_from if resumed else 0
        total = _announced_total(response, resumed)
        _update(bytes_done=offset, bytes_total=total)
        with open(part_path, "ab" if resumed else "wb") as part:
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                part.write(chunk)
                offset += len(chunk)
                _update(bytes_done=offset)
    if total is not None and offset != total:
        if offset > total:
            part_path.unlink()
        raise ResourceUnavailableError(_INCOMPLETE)


def _announced_total(response: httpx.Response, resumed: bool) -> int | None:
    if resumed:
        return _size_after_slash(response.headers.get("Content-Range"))
    length = response.headers.get("Content-Length")
    return int(length) if length is not None and length.isdigit() else None


def _size_after_slash(content_range: str | None) -> int | None:
    """The full size in a ``bytes 100-999/1000`` or ``bytes */1000`` header, else ``None``."""
    size = (content_range or "").rpartition("/")[2]
    return int(size) if size.isdigit() else None


def _failure_sentence(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return f"The model server answered with an error ({code}). Try again in a few minutes."
    if isinstance(exc, httpx.HTTPError):
        return _INTERRUPTED
    if isinstance(exc, OSError):
        return _DISK_FULL if exc.errno == errno.ENOSPC else _DISK_FAILED
    return _UNEXPECTED
