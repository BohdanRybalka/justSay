import asyncio
import sys
import threading
import time
import types

import pytest

from app.core.errors import ResourceUnavailableError
from app.stt.config import STTSettings
from app.stt.local import FASTER_WHISPER_REVISIONS, LocalSTTProvider, cached_model_folder
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


def _cache_folder(root, name, files=("model.bin", "config.json", "tokenizer.json")):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for file in files:
        (folder / file).write_bytes(b"x")
    return str(folder)


def _fake_faster_whisper(monkeypatch, tmp_path, whisper_model, download_model=None):
    calls: list[dict] = []

    def _download_model(size, local_files_only=False, revision=None):
        calls.append({"size": size, "local_files_only": local_files_only, "revision": revision})
        if download_model is not None:
            return download_model(size, local_files_only, revision)
        return _cache_folder(tmp_path, size)

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=whisper_model)
    )
    monkeypatch.setitem(
        sys.modules, "faster_whisper.utils", types.SimpleNamespace(download_model=_download_model)
    )
    return calls


@pytest.fixture
def built_models(monkeypatch, tmp_path):
    built: list[_FakeModel] = []
    hold = _Hold()

    def _whisper_model(size, device, compute_type):
        built.append(_FakeModel(size, hold if not built else None))
        return built[-1]

    _fake_faster_whisper(monkeypatch, tmp_path, _whisper_model)
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
    assert [model.size for model in built_models.models] == [str(tmp_path / "small")]

    built_models.hold.finish.set()
    await running
    assert reloaded.wait(3)
    reloading.join(3)

    assert [model.size for model in built_models.models] == [
        str(tmp_path / "small"),
        str(tmp_path / "medium"),
    ]
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
    monkeypatch, tmp_path
):
    building, finish_build = threading.Event(), threading.Event()

    def _whisper_model(size, device, compute_type):
        building.set()
        finish_build.wait(timeout=5)
        return _FakeModel(size)

    _fake_faster_whisper(monkeypatch, tmp_path, _whisper_model)
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


def _loader(loaded_from):
    return lambda path, device, compute_type: loaded_from.append(path) or _FakeModel(path)


def _cached_only_at(root, revisions):
    def _download_model(size, local_files_only, revision):
        if local_files_only and revision in revisions:
            return revisions[revision]
        if local_files_only:
            raise FileNotFoundError(size)
        return _cache_folder(root, f"downloaded-{size}")

    return _download_model


def test_a_model_cached_at_the_pinned_revision_loads_without_any_download(monkeypatch, tmp_path):
    loaded_from = []
    pinned = FASTER_WHISPER_REVISIONS["small"]
    folder = _cache_folder(tmp_path, "pinned")
    calls = _fake_faster_whisper(
        monkeypatch, tmp_path, _loader(loaded_from), _cached_only_at(tmp_path, {pinned: folder})
    )

    _provider()._get_model()

    assert loaded_from == [folder]
    assert calls == [{"size": "small", "local_files_only": True, "revision": pinned}]


def test_a_model_an_older_version_cached_under_main_loads_without_any_download(
    monkeypatch, tmp_path
):
    loaded_from = []
    folder = _cache_folder(tmp_path, "older")
    calls = _fake_faster_whisper(
        monkeypatch, tmp_path, _loader(loaded_from), _cached_only_at(tmp_path, {None: folder})
    )

    _provider()._get_model()

    assert loaded_from == [folder]
    assert all(call["local_files_only"] for call in calls)


def test_a_cached_folder_missing_a_file_is_downloaded_again_at_the_pinned_revision(
    monkeypatch, tmp_path
):
    loaded_from = []
    partial = _cache_folder(tmp_path, "partial", files=("config.json",))
    calls = _fake_faster_whisper(
        monkeypatch,
        tmp_path,
        _loader(loaded_from),
        _cached_only_at(tmp_path, {None: partial, FASTER_WHISPER_REVISIONS["small"]: partial}),
    )

    _provider()._get_model()

    assert loaded_from == [str(tmp_path / "downloaded-small")]
    assert calls[-1] == {
        "size": "small",
        "local_files_only": False,
        "revision": FASTER_WHISPER_REVISIONS["small"],
    }


def test_a_128_mel_model_without_its_feature_config_is_downloaded_again(monkeypatch, tmp_path):
    loaded_from = []
    incomplete = _cache_folder(tmp_path, "incomplete")
    calls = _fake_faster_whisper(
        monkeypatch,
        tmp_path,
        _loader(loaded_from),
        _cached_only_at(
            tmp_path, {None: incomplete, FASTER_WHISPER_REVISIONS["large-v3"]: incomplete}
        ),
    )
    provider = LocalSTTProvider(
        STTSettings(whisper_model_size="large-v3", whisper_device="cpu")
    )

    provider._get_model()

    assert loaded_from == [str(tmp_path / "downloaded-large-v3")]
    assert calls[-1]["local_files_only"] is False


def test_a_missing_faster_whisper_package_means_nothing_is_cached(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", None)

    assert cached_model_folder("small") is None


def test_a_missing_model_is_downloaded_once_at_its_pinned_revision(monkeypatch, tmp_path):
    loaded_from = []
    calls = _fake_faster_whisper(
        monkeypatch, tmp_path, _loader(loaded_from), _cached_only_at(tmp_path, {})
    )

    _provider()._get_model()

    assert loaded_from == [str(tmp_path / "downloaded-small")]
    assert [call["local_files_only"] for call in calls] == [True, True, False]
    assert calls[-1]["revision"] == FASTER_WHISPER_REVISIONS["small"]


def test_a_missing_model_of_an_unpinned_size_is_refused_without_a_download(
    monkeypatch, tmp_path
):
    calls = _fake_faster_whisper(
        monkeypatch, tmp_path, _loader([]), _cached_only_at(tmp_path, {})
    )
    provider = LocalSTTProvider(STTSettings(whisper_model_size="tiny.en", whisper_device="cpu"))

    with pytest.raises(ResourceUnavailableError, match="no verified download"):
        provider._get_model()

    assert all(call["local_files_only"] for call in calls)


def test_every_model_size_the_app_lists_has_a_pinned_faster_whisper_revision():
    from app.stt.local_setup import EXPECTED_MODEL_BYTES

    assert set(EXPECTED_MODEL_BYTES) == set(FASTER_WHISPER_REVISIONS)
    assert all(len(revision) == 40 for revision in FASTER_WHISPER_REVISIONS.values())
