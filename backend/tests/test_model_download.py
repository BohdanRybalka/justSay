"""Tests for `app.stt.model_download`: one resumable download, one shared progress state.

A real `httpx.Client` runs over a `MockTransport` that answers the way Hugging Face does:
`Range` requests get 206 with `Content-Range`, unless a test switches that off.
"""

import errno
import threading
import time

import httpx
import pytest

from app.core.errors import ResourceUnavailableError
from app.stt import model_download

DATA = bytes(range(256)) * 4
CHUNK = 64
REAL_CLIENT = httpx.Client


class _Server:
    def __init__(self, data=DATA, *, honours_range=True, break_after=None, status=None):
        self.data = data
        self.honours_range = honours_range
        self.break_after = break_after
        self.status = status
        self.range_headers = []
        self.progress_seen = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def handler(self, request):
        self.entered.set()
        self.release.wait(timeout=5)
        self.range_headers.append(request.headers.get("Range"))
        if self.status is not None:
            return httpx.Response(self.status)
        start = self._requested_start(request)
        if start is not None and start >= len(self.data):
            return httpx.Response(416, headers={"Content-Range": f"bytes */{len(self.data)}"})
        if start is None or not self.honours_range:
            headers = {"Content-Length": str(len(self.data))}
            return httpx.Response(200, headers=headers, stream=self._stream(0))
        headers = {"Content-Range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"}
        return httpx.Response(206, headers=headers, stream=self._stream(start))

    def _requested_start(self, request):
        header = request.headers.get("Range")
        return int(header.removeprefix("bytes=").rstrip("-")) if header else None

    def _stream(self, start):
        server = self

        class _Body(httpx.SyncByteStream):
            def __iter__(self):
                sent = 0
                for position in range(start, len(server.data), CHUNK):
                    server.progress_seen.append(model_download.snapshot().bytes_done)
                    if server.break_after is not None and sent >= server.break_after:
                        raise httpx.ReadError("connection lost")
                    piece = server.data[position : position + CHUNK]
                    sent += len(piece)
                    yield piece

        return _Body()


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(model_download, "_state", model_download.ModelDownloadState())
    monkeypatch.setattr(model_download, "_CHUNK_SIZE", CHUNK)
    yield
    assert not model_download._download_lock.locked()


def _serve(monkeypatch, server):
    monkeypatch.setattr(
        model_download.httpx,
        "Client",
        lambda **kwargs: REAL_CLIENT(transport=httpx.MockTransport(server.handler), **kwargs),
    )
    return server


def _paths(tmp_path):
    model_path = tmp_path / "models" / "ggml-tiny.bin"
    return model_path, model_path.with_name(model_path.name + ".part")


def test_download_writes_the_model_and_reports_every_byte(monkeypatch, tmp_path):
    server = _serve(monkeypatch, _Server())
    model_path, part_path = _paths(tmp_path)

    model_download.download("tiny", model_path)

    assert model_path.read_bytes() == DATA
    assert not part_path.exists()
    assert model_download.snapshot() == model_download.ModelDownloadState(
        stage="done", bytes_done=len(DATA), bytes_total=len(DATA)
    )
    assert server.range_headers == [None]


def test_progress_grows_chunk_by_chunk_while_the_transfer_runs(monkeypatch, tmp_path):
    server = _serve(monkeypatch, _Server())

    model_download.download("tiny", _paths(tmp_path)[0])

    assert server.progress_seen == list(range(0, len(DATA), CHUNK))


def test_an_interrupted_download_fails_in_plain_words_and_keeps_what_arrived(
    monkeypatch, tmp_path
):
    _serve(monkeypatch, _Server(break_after=CHUNK * 3))
    model_path, part_path = _paths(tmp_path)

    with pytest.raises(ResourceUnavailableError, match="interrupted"):
        model_download.download("tiny", model_path)

    state = model_download.snapshot()
    assert state.stage == "failed"
    assert "internet connection" in state.error
    assert part_path.read_bytes() == DATA[: CHUNK * 3]
    assert not model_path.exists()


def test_a_retry_resumes_from_the_partial_file_with_a_range_request(monkeypatch, tmp_path):
    model_path, part_path = _paths(tmp_path)
    broken = _serve(monkeypatch, _Server(break_after=CHUNK * 3))
    with pytest.raises(ResourceUnavailableError):
        model_download.download("tiny", model_path)

    healthy = _serve(monkeypatch, _Server())
    model_download.download("tiny", model_path)

    assert broken.range_headers == [None]
    assert healthy.range_headers == [f"bytes={CHUNK * 3}-"]
    assert model_path.read_bytes() == DATA
    assert not part_path.exists()
    state = model_download.snapshot()
    assert (state.stage, state.error) == ("done", None)


def test_a_server_that_ignores_the_range_restarts_from_zero(monkeypatch, tmp_path):
    model_path, part_path = _paths(tmp_path)
    part_path.parent.mkdir(parents=True)
    part_path.write_bytes(DATA[:CHUNK * 2])
    _serve(monkeypatch, _Server(honours_range=False))

    model_download.download("tiny", model_path)

    assert model_path.read_bytes() == DATA


def test_a_partial_file_longer_than_the_model_is_discarded_and_downloaded_again(
    monkeypatch, tmp_path
):
    model_path, part_path = _paths(tmp_path)
    part_path.parent.mkdir(parents=True)
    part_path.write_bytes(b"x" * (len(DATA) + 10))
    server = _serve(monkeypatch, _Server())

    model_download.download("tiny", model_path)

    assert model_path.read_bytes() == DATA
    assert server.range_headers == [f"bytes={len(DATA) + 10}-", None]


def test_a_partial_file_that_is_already_whole_is_only_renamed(monkeypatch, tmp_path):
    model_path, part_path = _paths(tmp_path)
    part_path.parent.mkdir(parents=True)
    part_path.write_bytes(DATA)
    server = _serve(monkeypatch, _Server())

    model_download.download("tiny", model_path)

    assert model_path.read_bytes() == DATA
    assert server.range_headers == [f"bytes={len(DATA)}-"]
    assert model_download.snapshot().bytes_total == len(DATA)


def test_a_body_shorter_than_announced_fails_and_keeps_the_partial_file(monkeypatch, tmp_path):
    server = _Server()
    server.handler = lambda request: httpx.Response(
        200, headers={"Content-Length": str(len(DATA) + 50)}, content=DATA
    )
    _serve(monkeypatch, server)
    model_path, part_path = _paths(tmp_path)

    with pytest.raises(ResourceUnavailableError, match="whole model"):
        model_download.download("tiny", model_path)

    assert part_path.read_bytes() == DATA
    assert not model_path.exists()


def test_a_server_error_names_its_status(monkeypatch, tmp_path):
    _serve(monkeypatch, _Server(status=404))

    with pytest.raises(ResourceUnavailableError, match=r"error \(404\)"):
        model_download.download("tiny", _paths(tmp_path)[0])

    assert "404" in model_download.snapshot().error


def test_a_location_that_cannot_be_written_says_so(monkeypatch, tmp_path):
    _serve(monkeypatch, _Server())
    blocker = tmp_path / "models"
    blocker.write_bytes(b"")

    with pytest.raises(ResourceUnavailableError, match="saved to the disk"):
        model_download.download("tiny", blocker / "ggml-tiny.bin")


def test_a_full_disk_is_named_as_such():
    sentence = model_download._failure_sentence(OSError(errno.ENOSPC, "No space left on device"))

    assert "not enough free space" in sentence


def test_download_returns_at_once_when_the_model_is_there(monkeypatch, tmp_path):
    server = _serve(monkeypatch, _Server())
    model_path, _ = _paths(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"already here")

    model_download.download("tiny", model_path)

    assert server.range_headers == []
    assert model_path.read_bytes() == b"already here"
    assert model_download.snapshot().stage == "idle"


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def test_start_answers_downloading_at_once_and_finishes_in_the_background(monkeypatch, tmp_path):
    server = _Server()
    server.release.clear()
    _serve(monkeypatch, server)
    model_path, _ = _paths(tmp_path)

    started = model_download.start("tiny", model_path)

    assert started.stage == "downloading"
    server.release.set()
    _wait_for(lambda: model_download.snapshot().stage == "done")
    assert model_path.read_bytes() == DATA
    _wait_for(lambda: not model_download._download_lock.locked())


def test_a_second_start_while_one_runs_does_not_start_another(monkeypatch, tmp_path):
    server = _Server()
    server.release.clear()
    _serve(monkeypatch, server)
    model_path, _ = _paths(tmp_path)
    model_download.start("tiny", model_path)
    assert server.entered.wait(timeout=5)

    again = model_download.start("tiny", model_path)

    assert again.stage == "downloading"
    server.release.set()
    _wait_for(lambda: model_download.snapshot().stage == "done")
    _wait_for(lambda: not model_download._download_lock.locked())
    assert len(server.range_headers) == 1


def test_a_load_that_needs_the_model_waits_for_the_download_install_started(
    monkeypatch, tmp_path
):
    server = _Server()
    server.release.clear()
    _serve(monkeypatch, server)
    model_path, _ = _paths(tmp_path)
    model_download.start("tiny", model_path)
    assert server.entered.wait(timeout=5)
    loader = threading.Thread(target=model_download.download, args=("tiny", model_path))
    loader.start()
    time.sleep(0.1)
    assert loader.is_alive()

    server.release.set()
    loader.join(timeout=5)

    assert not loader.is_alive()
    assert len(server.range_headers) == 1
    assert model_path.read_bytes() == DATA
    _wait_for(lambda: not model_download._download_lock.locked())


def test_start_does_nothing_when_the_model_is_there(monkeypatch, tmp_path):
    server = _serve(monkeypatch, _Server())
    model_path, _ = _paths(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"already here")

    state = model_download.start("tiny", model_path)

    assert state.stage == "idle"
    assert server.range_headers == []
    assert not model_download._download_lock.locked()


def test_a_failed_background_download_is_readable_from_the_state(monkeypatch, tmp_path):
    _serve(monkeypatch, _Server(break_after=CHUNK))

    model_download.start("tiny", _paths(tmp_path)[0])

    _wait_for(lambda: model_download.snapshot().stage == "failed")
    assert "interrupted" in model_download.snapshot().error
    _wait_for(lambda: not model_download._download_lock.locked())
