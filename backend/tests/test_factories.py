import pytest

from app.core.types import ProviderMode
from app.stt import clear_cache as clear_stt_cache
from app.stt import get_provider
from app.stt.cloud import GeminiSTTProvider
from app.stt.config import STTSettings
from app.stt.local import LocalSTTProvider
from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider


def test_stt_factory_returns_cloud_provider():
    clear_stt_cache()
    settings = STTSettings(mode=ProviderMode.CLOUD)
    provider = get_provider(settings.mode, settings)
    assert isinstance(provider, GeminiSTTProvider)


def test_stt_factory_returns_local_provider():
    """Proves only that LOCAL does not reach a cloud provider.

    The autouse `_force_faster_whisper_for_local` fixture pins
    `get_local_provider_class` to the very class this asserts, so a
    platform-routing regression cannot turn it red. The marked test below is
    the one that can -- JS-97.
    """
    clear_stt_cache()
    settings = STTSettings(mode=ProviderMode.LOCAL)
    provider = get_provider(settings.mode, settings)
    assert isinstance(provider, LocalSTTProvider)


@pytest.mark.no_factory_stub
def test_stt_factory_resolves_local_through_the_platform_factory(monkeypatch):
    """LOCAL returns whatever this platform's factory names, not a fixed class.

    Opted out of the autouse stub and pointed at the class macOS arm64 and
    Windows AMD/Intel actually get, so hard-coding any provider in
    `_get_local()` turns this red on every platform -- including a CI box whose
    real factory would have returned `LocalSTTProvider` anyway.
    """
    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: WhisperCppServerSTTProvider,
    )
    clear_stt_cache()
    settings = STTSettings(mode=ProviderMode.LOCAL)
    provider = get_provider(settings.mode, settings)
    assert type(provider) is WhisperCppServerSTTProvider
