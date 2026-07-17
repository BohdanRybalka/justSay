"""Tests for `app.stt.local_vulkan.WhisperCppVulkanSTTProvider`.

`subprocess.Popen` and `httpx.Client` are fully mocked -- no real process is
spawned and no real network call is made. `resolve_binary_path`/
`resolve_model_path` are monkeypatched per-test to point at `tmp_path`.
"""

import subprocess
import threading
import time

import pytest

import app.stt.local_vulkan as local_vulkan_module
from app.stt.config import STTSettings
from app.stt.local_vulkan import WhisperCppVulkanSTTProvider

# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class _FakeProcess:
    def __init__(self, exit_after_terminate: bool = True, wait_raises_timeout: bool = False):
        self.pid = 4242
        self.returncode = None
        self._exit_after_terminate = exit_after_terminate
        self._wait_raises_timeout = wait_raises_timeout
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self._exit_after_terminate:
            self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_raises_timeout:
            raise subprocess.TimeoutExpired(cmd="whisper-server", timeout=timeout)
        return self.returncode


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class _FakeStreamCtx:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, size):
        yield from self._chunks


class _FakeHttpxClient:
    def __init__(self, get_impl=None, post_impl=None, stream_impl=None, **kwargs):
        self._get_impl = get_impl
        self._post_impl = post_impl
        self._stream_impl = stream_impl

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        if self._get_impl is None:
            raise AssertionError("GET not expected in this test")
        return self._get_impl(url)

    def post(self, url, data=None, files=None):
        if self._post_impl is None:
            raise AssertionError("POST not expected in this test")
        return self._post_impl(url, data, files)

    def stream(self, method, url):
        if self._stream_impl is None:
            raise AssertionError("stream() not expected in this test")
        return self._stream_impl(method, url)


def _install_fake_httpx(monkeypatch, *, get_impl=None, post_impl=None, stream_impl=None):
    """Default `get_impl`: always healthy (200). Callers override for
    failure-path tests."""
    if get_impl is None:
        get_impl = lambda url: _FakeResponse(200)  # noqa: E731

    monkeypatch.setattr(
        local_vulkan_module.httpx,
        "Client",
        lambda *a, **kw: _FakeHttpxClient(
            get_impl=get_impl, post_impl=post_impl, stream_impl=stream_impl
        ),
    )


def _install_fake_popen(monkeypatch, *, exit_after_terminate: bool = True):
    calls: list = []
    process = _FakeProcess(exit_after_terminate=exit_after_terminate)

    def _fake_popen(argv, **kwargs):
        calls.append(argv)
        return process

    monkeypatch.setattr(local_vulkan_module.subprocess, "Popen", _fake_popen)
    return calls, process


def _make_provider(
    tmp_path, monkeypatch, *, model_exists: bool = True
) -> WhisperCppVulkanSTTProvider:
    binary_path = tmp_path / "whisper-server.exe"
    binary_path.write_bytes(b"")
    model_path = tmp_path / "ggml-large-v3-turbo.bin"
    if model_exists:
        model_path.write_bytes(b"fake-ggml-weights")

    monkeypatch.setattr(local_vulkan_module, "resolve_binary_path", lambda: binary_path)
    monkeypatch.setattr(local_vulkan_module, "resolve_model_path", lambda size: model_path)

    settings = STTSettings(whisper_model_size="large-v3-turbo")
    return WhisperCppVulkanSTTProvider(settings), model_path


# --------------------------------------------------------------------------- #
# Contract shape                                                              #
# --------------------------------------------------------------------------- #


def test_model_name_reflects_settings():
    settings = STTSettings(whisper_model_size="large-v3-turbo")
    provider = WhisperCppVulkanSTTProvider(settings)
    assert provider.model_name == "whisper-cpp-vulkan/large-v3-turbo"


def test_is_loaded_false_and_last_load_error_none_initially():
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    assert provider.is_loaded is False
    assert provider.last_load_error is None


def test_contract_shape_via_get_or_create(monkeypatch):
    """`app.stt.__init__`'s `is_model_loaded()`/`get_local_load_error()`/
    `clear_cache()` — all `getattr(provider, ...)`-based or a bare
    `provider.cleanup()` call — must work against this class with no code
    change in `app/stt/__init__.py`."""
    from app.stt import _get_or_create, get_local_load_error, is_model_loaded
    from app.stt import clear_cache as clear_stt_cache

    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: WhisperCppVulkanSTTProvider,
    )

    settings = STTSettings()
    provider = _get_or_create(WhisperCppVulkanSTTProvider, settings)

    assert is_model_loaded() is False
    assert get_local_load_error(settings) is None

    provider._server_ready = True
    assert is_model_loaded() is True

    provider._last_load_error = "boom"
    assert get_local_load_error(settings) == "boom"

    clear_stt_cache()  # must call provider.cleanup() without raising


# --------------------------------------------------------------------------- #
# _get_model — happy path, download, idempotency                             #
# --------------------------------------------------------------------------- #


def test_get_model_raises_and_latches_error_when_binary_missing(monkeypatch):
    monkeypatch.setattr(local_vulkan_module, "resolve_binary_path", lambda: None)
    provider = WhisperCppVulkanSTTProvider(STTSettings())

    with pytest.raises(RuntimeError):
        provider._get_model()

    assert provider.is_loaded is False
    assert "whisper-server binary not found" in provider.last_load_error


def test_get_model_downloads_missing_model_then_spawns_and_polls_healthy(monkeypatch, tmp_path):
    provider, model_path = _make_provider(tmp_path, monkeypatch, model_exists=False)
    assert not model_path.is_file()

    def _stream_impl(method, url):
        assert method == "GET"
        return _FakeStreamCtx([b"fake", b"-ggml-", b"weights"])

    _install_fake_httpx(monkeypatch, stream_impl=_stream_impl)
    popen_calls, _process = _install_fake_popen(monkeypatch)

    provider._get_model()

    assert model_path.is_file()
    assert model_path.read_bytes() == b"fake-ggml-weights"
    assert not model_path.with_name(model_path.name + ".part").exists()
    assert len(popen_calls) == 1
    assert provider.is_loaded is True
    assert provider.last_load_error is None


def test_get_model_does_not_redownload_existing_model(monkeypatch, tmp_path):
    provider, model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    original_bytes = model_path.read_bytes()

    _install_fake_httpx(monkeypatch)  # no stream_impl -- must not be called
    _install_fake_popen(monkeypatch)

    provider._get_model()

    assert model_path.read_bytes() == original_bytes
    assert provider.is_loaded is True


def test_get_model_latches_error_on_download_failure(monkeypatch, tmp_path):
    provider, model_path = _make_provider(tmp_path, monkeypatch, model_exists=False)

    def _stream_impl(method, url):
        return _FakeStreamCtx([], status_code=404)

    _install_fake_httpx(monkeypatch, stream_impl=_stream_impl)
    _install_fake_popen(monkeypatch)

    with pytest.raises(RuntimeError):
        provider._get_model()

    assert provider.is_loaded is False
    assert "404" in provider.last_load_error
    assert not model_path.is_file()


def test_get_model_latches_error_on_health_poll_timeout(monkeypatch, tmp_path):
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)

    _install_fake_httpx(monkeypatch, get_impl=lambda url: _FakeResponse(503))
    _install_fake_popen(monkeypatch)

    # Keep the test fast: shrink the poll budget and no-op the sleep between
    # attempts instead of waiting out the real ~120s budget.
    monkeypatch.setattr(local_vulkan_module, "_HEALTH_POLL_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(local_vulkan_module.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        provider._get_model()

    assert provider.is_loaded is False
    assert "did not become healthy" in provider.last_load_error


def test_get_model_terminates_orphaned_process_after_health_poll_timeout_then_retries(
    monkeypatch, tmp_path
):
    """Regression for Stage 3 review RED-2: `_spawn_server()` succeeding but
    `_wait_until_healthy()` timing out must not leak the spawned process --
    it's terminated and `_process` cleared so a subsequent `_get_model()`
    call is guaranteed to spawn a genuinely fresh process instead of
    silently overwriting a still-alive handle."""
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)

    _install_fake_httpx(monkeypatch, get_impl=lambda url: _FakeResponse(503))
    popen_calls, first_process = _install_fake_popen(monkeypatch)

    monkeypatch.setattr(local_vulkan_module, "_HEALTH_POLL_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(local_vulkan_module.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        provider._get_model()

    assert len(popen_calls) == 1
    assert first_process.terminate_calls == 1  # the orphaned process was terminated
    assert provider._process is None  # handle cleared -- nothing left to leak
    assert provider.is_loaded is False

    # Retry with a healthy server: must spawn a brand-new process, not reuse
    # or lose track of the leaked one.
    _install_fake_httpx(monkeypatch, get_impl=lambda url: _FakeResponse(200))
    second_process = _FakeProcess()

    def _fake_popen_2(argv, **kwargs):
        popen_calls.append(argv)
        return second_process

    monkeypatch.setattr(local_vulkan_module.subprocess, "Popen", _fake_popen_2)

    provider._get_model()

    assert len(popen_calls) == 2  # a genuinely fresh spawn, not a reused handle
    assert provider._process is second_process
    assert provider.is_loaded is True


def test_get_model_is_idempotent_when_already_warm(monkeypatch, tmp_path):
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    _install_fake_httpx(monkeypatch)
    popen_calls, _process = _install_fake_popen(monkeypatch)

    provider._get_model()
    assert len(popen_calls) == 1

    provider._get_model()  # second call -- must not respawn
    assert len(popen_calls) == 1
    assert provider.is_loaded is True


# --------------------------------------------------------------------------- #
# transcribe()                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_transcribe_spawns_server_at_most_once_across_two_calls(monkeypatch, tmp_path):
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    post_impl = lambda url, data, files: _FakeResponse(200, {"text": "hello\n"})  # noqa: E731
    _install_fake_httpx(monkeypatch, post_impl=post_impl)
    popen_calls, _process = _install_fake_popen(monkeypatch)

    result1 = await provider.transcribe(audio_path, language="uk")
    result2 = await provider.transcribe(audio_path, language="uk")

    assert len(popen_calls) == 1
    assert result1.text == "hello"
    assert result2.text == "hello"


@pytest.mark.asyncio
async def test_transcribe_joins_multiline_response_text(monkeypatch, tmp_path):
    """whisper-server's `output_str()` joins segments with `"\\n"` (no
    embedded timestamps) -- the provider collapses that to a single
    space-joined line, matching `LocalSTTProvider`'s output shape."""
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    post_impl = lambda url, data, files: _FakeResponse(  # noqa: E731
        200, {"text": "Привіт світ\nяк справи\n"}
    )
    _install_fake_httpx(monkeypatch, post_impl=post_impl)
    _install_fake_popen(monkeypatch)

    result = await provider.transcribe(audio_path, language="uk")

    assert result.text == "Привіт світ як справи"


@pytest.mark.asyncio
async def test_transcribe_sends_language_and_response_format(monkeypatch, tmp_path):
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    captured = {}

    def _post_impl(url, data, files):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        return _FakeResponse(200, {"text": "ok"})

    _install_fake_httpx(monkeypatch, post_impl=_post_impl)
    _install_fake_popen(monkeypatch)

    await provider.transcribe(audio_path, language="uk")

    assert captured["url"].endswith("/inference")
    assert captured["data"]["language"] == "uk"
    assert captured["data"]["response_format"] == "json"
    assert "file" in captured["files"]


# --------------------------------------------------------------------------- #
# cleanup()                                                                   #
# --------------------------------------------------------------------------- #


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll `predicate` until it's truthy or `timeout` elapses -- used below
    because the actual terminate()/kill() call now happens on a background
    daemon thread, so it must not be asserted on instantaneously."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_cleanup_returns_promptly_without_deadlock_when_load_lock_held():
    """Mirrors `LocalSTTProvider`'s/`MLXWhisperSTTProvider`'s established
    non-blocking-lock-guard test: `cleanup()` is reachable synchronously from
    `PUT /stt/mode`'s `clear_cache()` on the FastAPI event-loop thread, so it
    must never block on `_load_lock` for the duration of an in-flight load.
    """
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    sentinel_process = _FakeProcess()
    provider._process = sentinel_process
    provider._server_ready = True

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def _hold_lock():
        with provider._load_lock:
            lock_acquired.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=2), "holder thread never acquired the lock"

    start = time.monotonic()
    provider.cleanup()
    elapsed = time.monotonic() - start

    release_lock.set()
    holder.join(timeout=2)

    assert elapsed < 1.0, f"cleanup() blocked for {elapsed:.2f}s while the lock was held"
    assert provider._process is sentinel_process  # untouched — cleanup bailed out
    assert provider.is_loaded is True  # untouched — cleanup bailed out


def test_cleanup_returns_immediately_and_terminates_in_background_when_lock_is_free():
    """Regression for Stage 3 review issue #3: `cleanup()` must not block the
    FastAPI event loop for the grace-poll duration -- the bookkeeping
    (`_server_ready`, clearing `self._process`) stays synchronous, but the
    actual `.terminate()` call runs on a background daemon thread."""
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    process = _FakeProcess(exit_after_terminate=True)
    provider._process = process
    provider._server_ready = True

    start = time.monotonic()
    provider.cleanup()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"cleanup() blocked for {elapsed:.2f}s"
    assert provider._process is None  # cleared synchronously
    assert provider.is_loaded is False

    assert _wait_until(lambda: process.terminate_calls == 1), (
        "background thread never called terminate()"
    )
    assert process.kill_calls == 0


def test_cleanup_kills_process_in_background_when_terminate_does_not_exit_in_time(monkeypatch):
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    process = _FakeProcess(exit_after_terminate=False)
    provider._process = process
    provider._server_ready = True

    monkeypatch.setattr(local_vulkan_module, "_GRACE_POLL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(local_vulkan_module.time, "sleep", lambda _s: None)

    provider.cleanup()

    assert provider._process is None  # cleared synchronously, before the thread even runs

    assert _wait_until(lambda: process.kill_calls == 1), (
        "background thread never called kill()"
    )
    assert process.terminate_calls == 1


def test_cleanup_returns_immediately_even_though_terminate_sequence_would_block(monkeypatch):
    """Wall-clock regression: with a real (unmocked) grace-poll sleep,
    `cleanup()` itself must return in a small fraction of the time the
    underlying terminate() -> grace-poll -> kill() sequence actually takes --
    proof the sequence really runs off the calling thread, not just that the
    mocked sleep was skipped."""
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    process = _FakeProcess(exit_after_terminate=False)
    provider._process = process
    provider._server_ready = True

    monkeypatch.setattr(local_vulkan_module, "_GRACE_POLL_MAX_ATTEMPTS", 5)
    monkeypatch.setattr(local_vulkan_module, "_GRACE_POLL_INTERVAL", 0.05)
    grace_window = 5 * 0.05  # ~0.25s -- what a synchronous cleanup() would have blocked for

    start = time.monotonic()
    provider.cleanup()
    elapsed = time.monotonic() - start

    assert elapsed < grace_window / 2, (
        f"cleanup() blocked for {elapsed:.3f}s -- should return near-instantly"
    )

    assert _wait_until(lambda: process.kill_calls == 1, timeout=2.0), (
        "background thread never completed the full grace-poll + kill sequence"
    )
    assert process.terminate_calls == 1


def test_cleanup_is_a_noop_when_nothing_was_ever_spawned():
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    provider.cleanup()  # must not raise
    assert provider._process is None


# --------------------------------------------------------------------------- #
# _terminate_process() never raises (Stage 3 review, iteration 2, RED-2)     #
# --------------------------------------------------------------------------- #


def test_terminate_process_does_not_raise_when_kill_fallback_wait_times_out(monkeypatch):
    """`_terminate_process()` must return normally even when the process is
    still alive after both `.terminate()` and `.kill()` -- the `.kill()`
    fallback's `wait(timeout=3.0)` can raise `subprocess.TimeoutExpired`,
    which must be logged, not propagated."""
    provider = WhisperCppVulkanSTTProvider(STTSettings())
    process = _FakeProcess(exit_after_terminate=False, wait_raises_timeout=True)

    monkeypatch.setattr(local_vulkan_module, "_GRACE_POLL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(local_vulkan_module.time, "sleep", lambda _s: None)

    provider._terminate_process(process)  # must not raise

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1


def test_get_model_except_branch_reraises_original_error_not_timeout_expired(
    monkeypatch, tmp_path
):
    """Regression for Stage 3 review issue #2: when the health-poll times out
    AND the orphan-cleanup's kill fallback itself can't confirm the process
    died within its own 3s budget, `_get_model()`'s except-branch must still
    reach `self._process = None` and the caller must see the original
    health-poll-timeout `RuntimeError`, not a `TimeoutExpired` that replaced
    it."""
    provider, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)

    _install_fake_httpx(monkeypatch, get_impl=lambda url: _FakeResponse(503))

    process = _FakeProcess(exit_after_terminate=False, wait_raises_timeout=True)

    def _fake_popen(argv, **kwargs):
        return process

    monkeypatch.setattr(local_vulkan_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(local_vulkan_module, "_HEALTH_POLL_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(local_vulkan_module, "_GRACE_POLL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(local_vulkan_module.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="did not become healthy") as exc_info:
        provider._get_model()

    assert exc_info.type is RuntimeError  # not subprocess.TimeoutExpired
    assert process.kill_calls == 1  # the terminate -> kill sequence actually ran
    assert provider._process is None  # cleanup reached despite the wait() timeout
    assert provider.is_loaded is False


# --------------------------------------------------------------------------- #
# _port_lock cross-instance race (Stage 3 review, iteration 2, RED-1)        #
# --------------------------------------------------------------------------- #


class _BlockingTerminateProcess(_FakeProcess):
    """A `_FakeProcess` whose `.terminate()` blocks on a `threading.Event`
    before "dying" -- lets a test hold `_terminate_process()` inside its
    `_port_lock`-guarded section for as long as the test needs, and control
    exactly when it releases."""

    def __init__(self, started_event: threading.Event, release_event: threading.Event):
        super().__init__(exit_after_terminate=True)
        self._started_event = started_event
        self._release_event = release_event

    def terminate(self):
        self._started_event.set()
        self._release_event.wait(timeout=2)
        super().terminate()  # flips returncode -> "dead", same as the base fake


def test_port_lock_blocks_second_providers_spawn_until_first_providers_terminate_completes(
    monkeypatch, tmp_path
):
    """Regression for the exact reported race: rapid Local->Cloud->Local
    switching creates a brand-new provider instance (`clear_cache()` always
    replaces the cached provider) while the old instance's `cleanup()` may
    still be tearing down its `whisper-server` in a background thread. A
    second, unrelated provider's `_get_model()` must not spawn a new
    `whisper-server` on the shared fixed port until that in-flight
    termination has genuinely finished."""
    terminate_started = threading.Event()
    release_terminate = threading.Event()

    old_process = _BlockingTerminateProcess(terminate_started, release_terminate)
    provider1 = WhisperCppVulkanSTTProvider(STTSettings())

    # Exercise _terminate_process() directly on a background thread --
    # mirrors what cleanup()'s own background daemon thread does, without
    # depending on cleanup()'s lock-acquire bookkeeping for this test.
    terminate_thread = threading.Thread(
        target=provider1._terminate_process, args=(old_process,)
    )
    terminate_thread.start()
    assert terminate_started.wait(timeout=2), "first provider's terminate() never started"
    # _port_lock is now held by provider1's in-flight termination.

    provider2, _model_path = _make_provider(tmp_path, monkeypatch, model_exists=True)
    _install_fake_httpx(monkeypatch)  # health-poll always 200 once spawned
    popen_calls, _new_process = _install_fake_popen(monkeypatch)

    get_model_thread = threading.Thread(target=provider2._get_model)
    get_model_thread.start()

    # Give the second provider's thread every real chance to race ahead if
    # _port_lock weren't actually serializing the spawn against the
    # in-flight termination -- it must not have spawned yet.
    time.sleep(0.2)
    assert len(popen_calls) == 0, (
        "second provider's spawn proceeded before the first's in-flight "
        "termination completed -- _port_lock did not serialize them"
    )

    release_terminate.set()  # let the first termination finish and release _port_lock
    terminate_thread.join(timeout=2)
    assert not terminate_thread.is_alive()

    get_model_thread.join(timeout=2)
    assert not get_model_thread.is_alive()

    assert len(popen_calls) == 1
    assert provider2.is_loaded is True
    assert old_process.terminate_calls == 1


# --------------------------------------------------------------------------- #
# _download_lock cross-instance race (GitHub review, PR #21, iteration 1,     #
# issue #1)                                                                    #
# --------------------------------------------------------------------------- #


class _BlockingStreamCtx:
    """A `_FakeStreamCtx` whose `iter_bytes()` blocks on a `threading.Event`
    before yielding -- lets a test hold `_download_model()` inside its
    `_download_lock`-guarded section for as long as the test needs, mirroring
    `_BlockingTerminateProcess`'s Event-based approach for `_port_lock`
    above."""

    def __init__(self, chunks, started_event: threading.Event, release_event: threading.Event):
        self._chunks = chunks
        self._started_event = started_event
        self._release_event = release_event

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self, size):
        self._started_event.set()
        self._release_event.wait(timeout=2)
        yield from self._chunks


def test_download_lock_serializes_concurrent_downloads_and_second_skips_redownload(
    monkeypatch, tmp_path
):
    """Regression for the exact reported race: Spec 015's eager pre-warm
    starts a multi-GB download on a Local-mode switch; if the user switches
    away and back to Local within that window, `clear_cache()` hands out a
    second, independent `WhisperCppVulkanSTTProvider` instance whose own
    `_get_model()` sees the same model file missing and calls
    `_download_model()` too. `_download_lock` must serialize the two calls so
    neither writes to the shared `.part` file while the other is mid-stream,
    and the loser must skip the redundant download entirely once its
    post-lock `model_path.is_file()` recheck sees the winner already
    finished."""
    binary_path = tmp_path / "whisper-server.exe"
    binary_path.write_bytes(b"")
    model_path = tmp_path / "ggml-large-v3-turbo.bin"
    part_path = model_path.with_name(model_path.name + ".part")

    monkeypatch.setattr(local_vulkan_module, "resolve_binary_path", lambda: binary_path)
    monkeypatch.setattr(local_vulkan_module, "resolve_model_path", lambda size: model_path)

    settings = STTSettings(whisper_model_size="large-v3-turbo")
    provider1 = WhisperCppVulkanSTTProvider(settings)
    provider2 = WhisperCppVulkanSTTProvider(settings)

    download_started = threading.Event()
    release_download = threading.Event()
    stream_calls = {"n": 0}

    def _stream_impl(method, url):
        stream_calls["n"] += 1
        if stream_calls["n"] > 1:
            raise AssertionError(
                "a second _download_model() call started a redundant "
                "download instead of skipping it after the post-lock recheck"
            )
        return _BlockingStreamCtx(
            [b"fake", b"-ggml-", b"weights"], download_started, release_download
        )

    _install_fake_httpx(monkeypatch, stream_impl=_stream_impl)

    thread1 = threading.Thread(target=provider1._download_model, args=(model_path,))
    thread1.start()
    assert download_started.wait(timeout=2), "first download never started"
    # _download_lock is now held by provider1's in-flight download (the fake
    # stream is blocked mid-iter_bytes, inside the lock's `with` body).

    thread2 = threading.Thread(target=provider2._download_model, args=(model_path,))
    thread2.start()

    # Give the second call every real chance to race ahead if _download_lock
    # weren't actually serializing the two -- it must still be blocked, and
    # the model must not exist yet (the first download hasn't completed).
    time.sleep(0.2)
    assert thread2.is_alive(), (
        "second _download_model() call proceeded before the first released "
        "_download_lock"
    )
    assert not model_path.is_file()

    release_download.set()  # let the first download finish and release the lock
    thread1.join(timeout=2)
    assert not thread1.is_alive()

    thread2.join(timeout=2)
    assert not thread2.is_alive()

    assert model_path.is_file()
    assert model_path.read_bytes() == b"fake-ggml-weights"
    assert not part_path.exists()  # renamed on success, no leftover partial file
    assert stream_calls["n"] == 1  # second call skipped the redundant download
