import asyncio
import sys
import threading
import time
import types

import pytest

from app.core.errors import ResourceUnavailableError
from app.stt.config import STTSettings
from app.stt.local import FASTER_WHISPER_REVISIONS, LocalSTTProvider
from tests.conftest import wait_until


class _Info:
    language = "uk"


class _FakeModel:
    def __init__(self, size, hold=None):
        self.size = size
        self._hold = hold

    def transcribe(self, path, **kwargs):
        if self._hold is not None:
            self._hold.started.set()
            self._hold.finish.wait(timeout=5)
        return [], _Info()


class _Hold:
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()


def _fake_faster_whisper(monkeypatch, whisper_model, download_model=None):
    calls: list[dict] = []

    def _download_model(size, local_files_only=False, revision=None):
        calls.append({"size": size, "local_files_only": local_files_only, "revision": revision})
        if download_model is not None:
            return download_model(size, local_files_only, revision)
        return f"cache/{size}"

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=whisper_model)
    )
    monkeypatch.setitem(
        sys.modules, "faster_whisper.utils", types.SimpleNamespace(download_model=_download_model)
    )
    return calls


@pytest.fixture
def built_models(monkeypatch):
    built: list[_FakeModel] = []
    hold = _Hold()

    def _whisper_model(size, device, compute_type):
        built.append(_FakeModel(size, hold if not built else None))
        return built[-1]

    _fake_faster_whisper(monkeypatch, _whisper_model)
    built_hold = types.SimpleNamespace(models=built, hold=hold)
    return built_hold


def _provider() -> LocalSTTProvider:
    return LocalSTTProvider(STTSettings(whisper_model_size="small", whisper_device="cpu"))


@pytest.mark.asyncio
async def test_a_changed_model_never_sits_beside_the_one_a_request_is_using(
    built_models, tmp_path
):
    provider = _provider()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    running = asyncio.ensure_future(provider.transcribe(audio_path, language="uk"))
    assert await asyncio.to_thread(built_models.hold.started.wait, 2)
    provider._settings.whisper_model_size = "medium"
    reloaded = threading.Event()

    def _reload():
        provider._get_model()
        reloaded.set()

    reloading = threading.Thread(target=_reload)
    reloading.start()

    assert not reloaded.wait(0.3)
    assert [model.size for model in built_models.models] == ["cache/small"]

    built_models.hold.finish.set()
    await running
    assert reloaded.wait(3)
    reloading.join(3)

    assert [model.size for model in built_models.models] == ["cache/small", "cache/medium"]
    assert provider._model is built_models.models[1]


@pytest.mark.asyncio
async def test_a_changed_glossary_keeps_the_loaded_model(built_models, tmp_path):
    provider = _provider()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    built_models.hold.finish.set()

    await provider.transcribe(audio_path, language="uk")
    provider._settings.initial_prompt = "Tauri"
    await provider.transcribe(audio_path, language="uk")

    assert len(built_models.models) == 1


def test_a_model_made_for_other_settings_does_not_count_as_loaded(built_models):
    provider = _provider()
    provider._get_model()
    assert provider.is_loaded is True

    provider._settings.whisper_device = "cuda"

    assert provider.is_loaded is False


@pytest.mark.asyncio
async def test_a_release_waits_for_the_running_request_then_frees_the_model(
    built_models, tmp_path
):
    provider = _provider()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    running = asyncio.ensure_future(provider.transcribe(audio_path, language="uk"))
    assert await asyncio.to_thread(built_models.hold.started.wait, 2)

    provider.release_when_idle(lambda: True)
    assert provider.is_loaded is True

    built_models.hold.finish.set()
    await running

    assert wait_until(lambda: not provider.is_loaded)


def test_a_release_asked_for_during_a_load_is_dropped_if_a_route_uses_the_engine_by_its_end(
    monkeypatch,
):
    building, finish_build = threading.Event(), threading.Event()

    def _whisper_model(size, device, compute_type):
        building.set()
        finish_build.wait(timeout=5)
        return _FakeModel(size)

    _fake_faster_whisper(monkeypatch, _whisper_model)
    provider = _provider()
    loading = threading.Thread(target=provider._get_model)
    loading.start()
    assert building.wait(2)
    unused = [True]

    provider.release_when_idle(lambda: unused[0])
    unused[0] = False
    finish_build.set()
    loading.join(3)
    time.sleep(0.3)

    assert provider.is_loaded is True


def test_a_cached_model_loads_from_its_folder_without_any_download(monkeypatch):
    loaded_from = []
    calls = _fake_faster_whisper(
        monkeypatch, lambda path, device, compute_type: loaded_from.append(path) or _FakeModel(path)
    )

    _provider()._get_model()

    assert loaded_from == ["cache/small"]
    assert calls == [{"size": "small", "local_files_only": True, "revision": None}]


def test_a_missing_model_is_downloaded_once_at_its_pinned_revision(monkeypatch):
    loaded_from = []

    def _only_the_pinned_download_works(size, local_files_only, revision):
        if local_files_only:
            raise FileNotFoundError(size)
        return f"downloaded/{size}"

    calls = _fake_faster_whisper(
        monkeypatch,
        lambda path, device, compute_type: loaded_from.append(path) or _FakeModel(path),
        _only_the_pinned_download_works,
    )

    _provider()._get_model()

    assert loaded_from == ["downloaded/small"]
    assert calls[-1] == {
        "size": "small",
        "local_files_only": False,
        "revision": FASTER_WHISPER_REVISIONS["small"],
    }
    assert len(calls) == 2


def test_a_missing_model_of_an_unpinned_size_is_refused_without_a_download(monkeypatch):
    def _nothing_cached(size, local_files_only, revision):
        raise FileNotFoundError(size)

    calls = _fake_faster_whisper(
        monkeypatch, lambda *args, **kwargs: _FakeModel("x"), _nothing_cached
    )
    provider = LocalSTTProvider(STTSettings(whisper_model_size="tiny.en", whisper_device="cpu"))

    with pytest.raises(ResourceUnavailableError, match="no verified download"):
        provider._get_model()

    assert [call["local_files_only"] for call in calls] == [True]


def test_every_model_size_the_app_lists_has_a_pinned_faster_whisper_revision():
    from app.stt.local_setup import EXPECTED_MODEL_BYTES

    assert set(EXPECTED_MODEL_BYTES) == set(FASTER_WHISPER_REVISIONS)
    assert all(len(revision) == 40 for revision in FASTER_WHISPER_REVISIONS.values())
