from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.llm import get_llm_provider, clear_cache
from app.llm.config import LLMSettings
from app.llm.cloud import CloudLLMProvider
from app.llm.local import LocalLLMProvider


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    clear_cache()
    yield
    clear_cache()


# --- Factory caching ---


def test_factory_caches_provider():
    settings = LLMSettings(mode=ProviderMode.CLOUD)
    p1 = get_llm_provider(settings)
    p2 = get_llm_provider(settings)
    assert p1 is p2


def test_factory_invalidates_on_mode_change():
    cloud_settings = LLMSettings(mode=ProviderMode.CLOUD)
    local_settings = LLMSettings(mode=ProviderMode.LOCAL)
    p1 = get_llm_provider(cloud_settings)
    p2 = get_llm_provider(local_settings)
    assert p1 is not p2
    assert isinstance(p1, CloudLLMProvider)
    assert isinstance(p2, LocalLLMProvider)


# --- Cloud LLM ---


def test_cloud_llm_model_name():
    settings = LLMSettings(mode=ProviderMode.CLOUD, groq_model="llama-4-scout")
    provider = CloudLLMProvider(settings)
    assert provider.model_name == "groq/llama-4-scout"


def test_cloud_llm_requires_api_key():
    settings = LLMSettings(mode=ProviderMode.CLOUD, groq_api_key="")
    provider = CloudLLMProvider(settings)
    with pytest.raises(RuntimeError, match="JUSTSAY_LLM_GROQ_API_KEY"):
        provider._get_client()


@pytest.mark.asyncio
async def test_cloud_llm_process():
    settings = LLMSettings(mode=ProviderMode.CLOUD, groq_api_key="test-key")
    provider = CloudLLMProvider(settings)
    provider._client = MagicMock()  # skip _get_client

    with patch.object(CloudLLMProvider, "_call_groq", return_value="  Cleaned text  "):
        result = await provider.process("raw text", system_prompt="clean it")

    assert result == "Cleaned text"


@pytest.mark.asyncio
async def test_cloud_llm_returns_empty_for_none():
    settings = LLMSettings(mode=ProviderMode.CLOUD, groq_api_key="test-key")
    provider = CloudLLMProvider(settings)
    provider._client = MagicMock()

    with patch.object(CloudLLMProvider, "_call_groq", return_value=None):
        result = await provider.process("raw text", system_prompt="clean it")

    assert result == ""


# --- Local LLM ---


def test_local_llm_model_name():
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    assert provider.model_name == "ollama/qwen3:1.7b"


@pytest.mark.asyncio
async def test_local_llm_process():
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    provider._client = MagicMock()  # skip _get_client

    with patch.object(LocalLLMProvider, "_call_ollama", return_value="  cleaned  "):
        result = await provider.process("raw", system_prompt="clean")

    assert result == "cleaned"


def test_local_llm_cleanup_clears_client():
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is None
