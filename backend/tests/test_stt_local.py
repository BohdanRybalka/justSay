import asyncio
import sys
import threading
import types

import pytest

from app.stt.config import STTSettings
from app.stt.local import LocalSTTProvider
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


@pytest.fixture
def built_models(monkeypatch):
    built: list[_FakeModel] = []
    hold = _Hold()

    def _whisper_model(size, device, compute_type):
        built.append(_FakeModel(size, hold if not built else None))
        return built[-1]

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_whisper_model)
    )
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
    assert [model.size for model in built_models.models] == ["small"]

    built_models.hold.finish.set()
    await running
    assert reloaded.wait(3)
    reloading.join(3)

    assert [model.size for model in built_models.models] == ["small", "medium"]
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
