"""Tests for `app.stt.local_mlx.MLXWhisperSTTProvider`.

Every test is marked `@pytest.mark.mlx` so the autouse
`_force_faster_whisper_for_local` fixture in `conftest.py` opts out and the
real `MLXWhisperSTTProvider` class is exercised. None of these tests require
the `mlx-whisper` package to be installed — `mlx_whisper.transcribe` and
`mlx_whisper.load_models` are stubbed via `sys.modules` injection.
"""

import asyncio
import os
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.mlx




class _FakeModelHolder:
    """Stand-in for `mlx_whisper.transcribe.ModelHolder`. Class-var pattern."""

    model = None
    model_path = None


def _install_mlx_whisper_stub(monkeypatch, *, load_model=None, transcribe=None):
    """Inject minimal `mlx_whisper` / `mlx_whisper.load_models` /
    `mlx_whisper.transcribe` modules into `sys.modules` plus a `mlx.core`
    stub for cleanup tests.

    Each call resets `_FakeModelHolder` so tests don't share state.
    """
    _FakeModelHolder.model = None
    _FakeModelHolder.model_path = None

    transcribe_mod = types.ModuleType("mlx_whisper.transcribe")
    transcribe_mod.ModelHolder = _FakeModelHolder

    load_models_mod = types.ModuleType("mlx_whisper.load_models")
    load_models_mod.load_model = load_model or (lambda repo_id: None)

    pkg = types.ModuleType("mlx_whisper")
    pkg.transcribe = transcribe or (
        lambda audio_path, **kw: {"text": "stub", "segments": []}
    )
    pkg.load_models = load_models_mod

    monkeypatch.setitem(sys.modules, "mlx_whisper", pkg)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", transcribe_mod)
    monkeypatch.setitem(sys.modules, "mlx_whisper.load_models", load_models_mod)

    mx_metal = types.ModuleType("mlx.core.metal")
    mx_metal.clear_cache = MagicMock()
    mx_core = types.ModuleType("mlx.core")
    mx_core.metal = mx_metal
    mx = types.ModuleType("mlx")
    mx.core = mx_core
    monkeypatch.setitem(sys.modules, "mlx", mx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx_core)
    monkeypatch.setitem(sys.modules, "mlx.core.metal", mx_metal)

    return pkg, load_models_mod, transcribe_mod


def _settings(size: str = "large-v3-turbo"):
    from app.core.types import ProviderMode
    from app.stt.config import STTSettings

    return STTSettings(mode=ProviderMode.LOCAL, whisper_model_size=size)


def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)




def test_repo_map_has_all_seven_supported_sizes():
    from app.stt.local_mlx import MLX_REPO_BY_SIZE

    expected = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    assert MLX_REPO_BY_SIZE == expected


def test_repo_id_resolution_for_default_size():
    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    assert provider._get_repo_id() == "mlx-community/whisper-large-v3-turbo"


def test_repo_id_unknown_size_raises_value_error():
    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v4-quad-turbo"))
    with pytest.raises(ValueError, match="large-v4-quad-turbo"):
        provider._get_repo_id()




def test_get_model_unknown_size_latches_error_into_last_load_error(monkeypatch):
    _clean_env(monkeypatch)
    _install_mlx_whisper_stub(monkeypatch)

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("does-not-exist"))
    with pytest.raises(ValueError):
        provider._get_model()
    assert provider.last_load_error is not None
    assert "does-not-exist" in provider.last_load_error


def test_get_model_unknown_size_latches_even_when_model_is_warm(monkeypatch):
    """User mid-stream changes whisper_model_size to garbage with a model
    already in `ModelHolder.model` — `last_load_error` must still update."""
    _clean_env(monkeypatch)
    _install_mlx_whisper_stub(monkeypatch)
    _FakeModelHolder.model = object()
    _FakeModelHolder.model_path = "mlx-community/whisper-large-v3-turbo"

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("bogus"))
    with pytest.raises(ValueError):
        provider._get_model()
    assert provider.last_load_error is not None and "bogus" in provider.last_load_error


def test_get_model_short_circuits_when_already_warm_for_same_repo(monkeypatch):
    _clean_env(monkeypatch)
    load_model_spy = MagicMock()
    _install_mlx_whisper_stub(monkeypatch, load_model=load_model_spy)
    _FakeModelHolder.model = object()
    _FakeModelHolder.model_path = "mlx-community/whisper-large-v3-turbo"

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    provider._get_model()
    load_model_spy.assert_not_called()


def test_get_model_does_not_short_circuit_when_cached_repo_differs(monkeypatch):
    _clean_env(monkeypatch)
    load_model_spy = MagicMock()
    _install_mlx_whisper_stub(monkeypatch, load_model=load_model_spy)
    _FakeModelHolder.model = object()
    _FakeModelHolder.model_path = "mlx-community/whisper-large-v3-mlx"

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    provider._get_model()
    load_model_spy.assert_called_once_with("mlx-community/whisper-large-v3-turbo")


def test_get_model_uses_scan_cache_dir_helper(monkeypatch):
    """`_hf_cache_has_snapshot` must consult `app.stt.local_mlx.scan_cache_dir`.

    Patching the local-module-bound symbol (not `huggingface_hub.scan_cache_dir`)
    is the correct target because the file does
    `from huggingface_hub import scan_cache_dir`.
    """
    _clean_env(monkeypatch)
    _install_mlx_whisper_stub(monkeypatch)
    scan_spy = MagicMock(return_value=MagicMock(repos=[]))
    monkeypatch.setattr("app.stt.local_mlx.scan_cache_dir", scan_spy)

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    provider._get_model()
    scan_spy.assert_called()


def test_get_model_sets_hf_hub_offline_when_cache_present(monkeypatch):
    _clean_env(monkeypatch)

    def _load_model(repo_id):
        _load_model.observed = os.environ.get("HF_HUB_OFFLINE")

    _load_model.observed = None
    _install_mlx_whisper_stub(monkeypatch, load_model=_load_model)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(
            return_value=MagicMock(
                repos=[MagicMock(repo_id="mlx-community/whisper-large-v3-turbo")]
            )
        ),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    provider._get_model()
    assert _load_model.observed == "1"


def test_get_model_does_not_set_hf_hub_offline_when_cache_empty(monkeypatch):
    _clean_env(monkeypatch)

    def _load_model(repo_id):
        _load_model.observed = os.environ.get("HF_HUB_OFFLINE")

    _load_model.observed = "untouched"
    _install_mlx_whisper_stub(monkeypatch, load_model=_load_model)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))
    provider._get_model()
    assert _load_model.observed is None


def test_get_model_pops_hf_hub_offline_when_switching_to_uncached_model(monkeypatch):
    """Switch-model-mid-session: model A cached (flag=1) → model B not cached.

    The flag must be popped before `load_model(B)` is called, otherwise the
    first download of B fails with OfflineModeIsEnabled.
    """
    _clean_env(monkeypatch)
    os.environ["HF_HUB_OFFLINE"] = "1"

    def _load_model(repo_id):
        _load_model.observed = os.environ.get("HF_HUB_OFFLINE")

    _load_model.observed = "untouched"
    _install_mlx_whisper_stub(monkeypatch, load_model=_load_model)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3"))
    provider._get_model()
    assert _load_model.observed is None




def test_load_lock_serialises_concurrent_get_model(monkeypatch):
    """Two near-simultaneous `_get_model` calls must funnel through the same
    sync lock; load_model is invoked exactly once."""
    _clean_env(monkeypatch)
    call_count = {"n": 0}

    def _load_model(repo_id):
        call_count["n"] += 1
        time.sleep(0.05)
        _FakeModelHolder.model = object()
        _FakeModelHolder.model_path = repo_id

    _install_mlx_whisper_stub(monkeypatch, load_model=_load_model)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings("large-v3-turbo"))

    threads = [threading.Thread(target=provider._get_model) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count["n"] == 1




def test_is_loaded_starts_false(monkeypatch):
    _install_mlx_whisper_stub(monkeypatch)
    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    assert provider.is_loaded is False


def test_is_loaded_stays_false_when_transcribe_fails(monkeypatch, tmp_path):
    _clean_env(monkeypatch)

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    _install_mlx_whisper_stub(monkeypatch, transcribe=_boom)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(provider.transcribe(audio, language="uk"))
    assert provider.is_loaded is False


def test_is_loaded_flips_to_true_after_successful_transcribe(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    _install_mlx_whisper_stub(
        monkeypatch,
        transcribe=lambda audio, **kw: {"text": "hello", "segments": []},
    )
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")
    result = asyncio.run(provider.transcribe(audio, language="uk"))
    assert result.text == "hello"
    assert provider.is_loaded is True


def test_cleanup_resets_model_holder_and_clears_metal_cache(monkeypatch):
    _install_mlx_whisper_stub(monkeypatch)
    _FakeModelHolder.model = object()
    _FakeModelHolder.model_path = "mlx-community/whisper-large-v3-turbo"

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    provider._loaded = True
    provider.cleanup()

    assert _FakeModelHolder.model is None
    assert _FakeModelHolder.model_path is None
    assert provider.is_loaded is False
    sys.modules["mlx.core.metal"].clear_cache.assert_called_once()


def test_cleanup_returns_promptly_without_deadlock_when_load_lock_held(monkeypatch):
    """`cleanup()` must not block when `_load_lock` is already held by an
    in-flight `_get_model()` call on another thread — critical because
    `cleanup()` runs synchronously on the FastAPI event-loop thread via
    `clear_cache()` (spec 015, RED-3, iteration 2 triage)."""
    _install_mlx_whisper_stub(monkeypatch)
    _FakeModelHolder.model = object()
    _FakeModelHolder.model_path = "mlx-community/whisper-large-v3-turbo"

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    provider._loaded = True

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
    assert _FakeModelHolder.model is not None
    assert provider.is_loaded is True




def test_short_audio_kwargs_temperature_zero_no_vad_no_ngram(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    captured_kwargs: dict = {}

    def _transcribe(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    settings = _settings()
    provider = MLXWhisperSTTProvider(settings)
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    asyncio.run(
        provider.transcribe(audio, language="uk", audio_duration=5.0)
    )

    assert captured_kwargs["beam_size"] == 1
    assert captured_kwargs["condition_on_previous_text"] is False
    assert captured_kwargs["temperature"] == 0.0
    assert "no_repeat_ngram_size" not in captured_kwargs
    assert "vad_filter" not in captured_kwargs


def test_long_audio_kwargs_no_temperature_pin(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    captured_kwargs: dict = {}

    def _transcribe(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    asyncio.run(
        provider.transcribe(audio, language="uk", audio_duration=120.0)
    )

    assert captured_kwargs["beam_size"] == 5
    assert captured_kwargs["condition_on_previous_text"] is True
    assert "temperature" not in captured_kwargs


def test_transcribe_lock_serialises_concurrent_calls(monkeypatch, tmp_path):
    """Two concurrent `transcribe()` awaits — the inner sync `_run_mlx`
    must never be re-entered while the first call is still in flight."""
    _clean_env(monkeypatch)

    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def _transcribe(audio_path, **kwargs):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    async def _both():
        await asyncio.gather(
            provider.transcribe(audio, language="uk", audio_duration=5.0),
            provider.transcribe(audio, language="uk", audio_duration=5.0),
        )

    asyncio.run(_both())
    assert active["max"] == 1




def test_explicit_language_passed_through_unchanged(monkeypatch, tmp_path):
    """Regression: an explicit BCP-47 code must still reach mlx_whisper.transcribe as-is."""
    _clean_env(monkeypatch)
    captured_kwargs: dict = {}

    def _transcribe(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    asyncio.run(provider.transcribe(audio, language="uk", audio_duration=5.0))

    assert captured_kwargs["language"] == "uk"


def test_detected_language_populated_from_result_language_key(monkeypatch, tmp_path):
    """AC-17 (spec 029): mlx-whisper's result dict `"language"` key reaches
    TranscriptionResult.detected_language, normalized -- populated whether
    or not `language` was "auto"."""
    _clean_env(monkeypatch)

    def _transcribe(audio_path, **kwargs):
        return {"text": "hello", "segments": [], "language": "en"}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    result = asyncio.run(provider.transcribe(audio, language="auto"))

    assert result.detected_language == "en"


def test_detected_language_none_when_result_has_no_language_key(monkeypatch, tmp_path):
    _clean_env(monkeypatch)

    def _transcribe(audio_path, **kwargs):
        return {"text": "hello", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    result = asyncio.run(provider.transcribe(audio, language="uk"))

    assert result.detected_language is None


def test_auto_language_translates_to_none(monkeypatch, tmp_path):
    """language="auto" must become language=None -- mlx-whisper's own native
    auto-detect sentinel (same convention as faster-whisper), not the literal
    string "auto"."""
    _clean_env(monkeypatch)
    captured_kwargs: dict = {}

    def _transcribe(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    asyncio.run(provider.transcribe(audio, language="auto", audio_duration=5.0))

    assert captured_kwargs["language"] is None


def test_event_loop_not_blocked_during_get_model(monkeypatch, tmp_path):
    """`_run_mlx` sleeping in a worker thread must NOT freeze the event loop.

    Spawns a concurrent `asyncio.sleep(0.01)` and asserts both complete; if
    `_get_model` ever ran on the loop thread, the loop would have stalled.
    """
    _clean_env(monkeypatch)

    def _slow_transcribe(audio_path, **kwargs):
        time.sleep(0.1)
        return {"text": "ok", "segments": []}

    _install_mlx_whisper_stub(monkeypatch, transcribe=_slow_transcribe)
    monkeypatch.setattr(
        "app.stt.local_mlx.scan_cache_dir",
        MagicMock(return_value=MagicMock(repos=[])),
    )

    from app.stt.local_mlx import MLXWhisperSTTProvider

    provider = MLXWhisperSTTProvider(_settings())
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF....WAVEfake")

    ticks = {"n": 0}

    async def _tick():
        for _ in range(20):
            await asyncio.sleep(0.005)
            ticks["n"] += 1

    async def _both():
        await asyncio.gather(
            provider.transcribe(audio, language="uk", audio_duration=5.0),
            _tick(),
        )

    asyncio.run(_both())
    assert ticks["n"] >= 10
