from app.core.types import ProviderMode
from app.stt import clear_cache as clear_stt_cache, get_provider
from app.stt.config import STTSettings
from app.stt.cloud import GeminiSTTProvider
from app.stt.local import LocalSTTProvider
from app.llm import get_llm_provider, clear_cache as clear_llm_cache
from app.llm.config import LLMSettings
from app.llm.cloud import CloudLLMProvider
from app.llm.local import LocalLLMProvider


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


def test_llm_factory_returns_cloud_provider():
    clear_llm_cache()
    settings = LLMSettings(mode=ProviderMode.CLOUD)
    provider = get_llm_provider(settings)
    assert isinstance(provider, CloudLLMProvider)


def test_llm_factory_returns_local_provider():
    clear_llm_cache()
    settings = LLMSettings(mode=ProviderMode.LOCAL)
    provider = get_llm_provider(settings)
    assert isinstance(provider, LocalLLMProvider)
