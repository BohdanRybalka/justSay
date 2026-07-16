import logging
from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.llm import get_llm_provider, clear_cache
from app.llm.config import LLMSettings
from app.llm.cloud import CloudLLMProvider
from app.llm.local import LocalLLMProvider
from app.llm.tasks import DEFAULT_TASK, TASK_PROFILES, get_task_profile


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
    with pytest.raises(RuntimeError, match="missing"):
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


@pytest.mark.asyncio
async def test_cloud_llm_process_forwards_task_profile():
    """`process(task=...)` must resolve the profile and forward its
    temperature/top_p/max_tokens to `_call_groq` — not the old hardcoded
    0.1/4096. Compared against `TASK_PROFILES` directly so the two can't drift."""
    settings = LLMSettings(mode=ProviderMode.CLOUD, groq_api_key="test-key")
    provider = CloudLLMProvider(settings)
    provider._client = MagicMock()

    with patch.object(CloudLLMProvider, "_call_groq", return_value="structured") as mock_call:
        await provider.process("raw text", system_prompt="structure it", task="ai_prompt_structuring")

    profile = TASK_PROFILES["ai_prompt_structuring"]
    assert mock_call.call_args.kwargs["temperature"] == profile.temperature
    assert mock_call.call_args.kwargs["top_p"] == profile.top_p
    assert mock_call.call_args.kwargs["max_tokens"] == profile.max_tokens


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


@pytest.mark.asyncio
async def test_local_llm_process_forwards_task_profile_and_disables_thinking():
    """The real `_call_ollama` must pass `think=False` unconditionally and
    the task-resolved temperature/top_p/num_predict to the Ollama SDK client —
    mirroring the Cloud-side profile-forwarding test above, one level deeper
    since `think=False` and the warning-log logic live inside `_call_ollama`
    itself, not in `process()`."""
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "structured text", "thinking": ""}}
    provider._client = mock_client

    await provider.process("raw", system_prompt="structure it", task="ai_prompt_structuring")

    profile = TASK_PROFILES["ai_prompt_structuring"]
    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["think"] is False
    assert call_kwargs["options"]["temperature"] == profile.temperature
    assert call_kwargs["options"]["top_p"] == profile.top_p
    assert call_kwargs["options"]["num_predict"] == profile.max_tokens


@pytest.mark.asyncio
async def test_local_llm_warns_when_thinking_starves_content(caplog):
    """Defensive signal: an Ollama server/model combo that ignores
    `think=False` and returns only reasoning text must log a warning, not
    raise, and `process()`'s empty-string contract must hold."""
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {"content": "", "thinking": "some reasoning text"}
    }
    provider._client = mock_client

    with caplog.at_level(logging.WARNING, logger="app.llm.local"):
        result = await provider.process("raw", system_prompt="structure it")

    assert result == ""
    assert any(
        "thinking" in rec.getMessage().lower() and "empty content" in rec.getMessage().lower()
        for rec in caplog.records
    )


def test_local_llm_cleanup_clears_client():
    settings = LLMSettings(mode=ProviderMode.LOCAL, ollama_model="qwen3:1.7b")
    provider = LocalLLMProvider(settings)
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is None


# --- Task generation profiles ---


def test_task_profiles_has_exactly_three_entries():
    assert set(TASK_PROFILES) == {"dictation_cleanup", "ai_prompt_structuring", "insights"}


def test_get_task_profile_known_task():
    assert get_task_profile("insights") == TASK_PROFILES["insights"]


def test_get_task_profile_unknown_task_falls_back_to_default():
    """Fail-soft: an unrecognized task name must not raise — it resolves to
    the default task's profile, same as `get_task_profile`'s own docstring
    promises for loose-string callers (HTTP `style` mapping, etc.)."""
    assert get_task_profile("not_a_real_task") == TASK_PROFILES[DEFAULT_TASK]
