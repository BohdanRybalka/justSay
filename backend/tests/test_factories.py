from app.core.types import ProviderMode
from app.stt import clear_cache as clear_stt_cache
from app.stt import get_provider
from app.stt.cloud import GeminiSTTProvider
from app.stt.config import STTSettings
from app.stt.local import LocalSTTProvider


def test_stt_factory_returns_cloud_provider():
    clear_stt_cache()
    settings = STTSettings(mode=ProviderMode.CLOUD)
    provider = get_provider(settings.mode, settings)
    assert isinstance(provider, GeminiSTTProvider)


def test_stt_factory_returns_local_provider():
    clear_stt_cache()
    settings = STTSettings(mode=ProviderMode.LOCAL)
    provider = get_provider(settings.mode, settings)
    assert isinstance(provider, LocalSTTProvider)
