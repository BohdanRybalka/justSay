from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.stt.config import STTSettings
from app.stt.groq_whisper import GroqWhisperSTTProvider


def _settings(**overrides) -> STTSettings:
    defaults = dict(groq_api_key="test-key", groq_whisper_model="whisper-large-v3-turbo")
    defaults.update(overrides)
    return STTSettings(**defaults)


def _wav(tmp_path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    p = tmp_path / "s.wav"
    sf.write(str(p), np.zeros(16000, dtype=np.float32), 16000)
    return p


def test_model_name_uses_configured_model():
    provider = GroqWhisperSTTProvider(_settings(groq_whisper_model="whisper-custom"))
    assert provider.model_name == "groq/whisper-custom"


def test_missing_api_key_raises():
    provider = GroqWhisperSTTProvider(_settings(groq_api_key=""))
    with pytest.raises(RuntimeError, match="JUSTSAY_STT_GROQ_API_KEY"):
        provider._get_client()


@pytest.mark.asyncio
async def test_transcribe_returns_stripped_text(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()  # skip real SDK init

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value="  привіт  "):
        result = await provider.transcribe(_wav(tmp_path), language="uk")

    assert result.text == "привіт"


@pytest.mark.asyncio
async def test_transcribe_ignores_unknown_kwargs(tmp_path):
    """style=ai_prompt must not crash Groq — it just ignores it."""
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value="ok"):
        result = await provider.transcribe(_wav(tmp_path), language="uk", style="ai_prompt")

    assert result.text == "ok"


def test_rate_limit_raises_clearer_runtime_error(tmp_path):
    """_call_groq must translate HTTP 429 into a RuntimeError with helpful text."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = Exception(
        "HTTP 429: rate_limit_exceeded"
    )

    with pytest.raises(RuntimeError, match="Groq rate limit"):
        provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk")


def test_other_errors_bubble_up_unchanged(tmp_path):
    """Non-429 SDK errors must propagate as-is so callers see the root cause."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = ValueError("invalid audio")

    with pytest.raises(ValueError, match="invalid audio"):
        provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk")


def test_cleanup_resets_client():
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is None
