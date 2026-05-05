from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from app.core.types import ProviderMode
from app.stt import clear_cache, get_provider
from app.stt.config import STTSettings
from app.stt.cloud import GeminiSTTProvider
from app.stt.local import LocalSTTProvider


@pytest.fixture(autouse=True)
def _clear_stt_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def sample_wav(tmp_path) -> Path:
    """Create a short test WAV file."""
    audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)  # 1 second
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, 16000)
    return path


# --- Factory caching ---


def test_factory_caches_provider():
    settings = STTSettings(mode=ProviderMode.CLOUD)
    p1 = get_provider(settings.mode, settings)
    p2 = get_provider(settings.mode, settings)
    assert p1 is p2


def test_factory_invalidates_on_mode_change():
    cloud_settings = STTSettings(mode=ProviderMode.CLOUD)
    local_settings = STTSettings(mode=ProviderMode.LOCAL)
    p1 = get_provider(cloud_settings.mode, cloud_settings)
    p2 = get_provider(local_settings.mode, local_settings)
    assert p1 is not p2
    assert isinstance(p1, GeminiSTTProvider)
    assert isinstance(p2, LocalSTTProvider)


# --- Cloud STT ---


def test_cloud_stt_model_name():
    settings = STTSettings(mode=ProviderMode.CLOUD, gemini_model="gemini-2.5-flash")
    provider = GeminiSTTProvider(settings)
    assert provider.model_name == "gemini/gemini-2.5-flash"


def test_cloud_stt_requires_api_key():
    settings = STTSettings(mode=ProviderMode.CLOUD, gemini_api_key="")
    provider = GeminiSTTProvider(settings)
    with pytest.raises(RuntimeError, match="JUSTSAY_STT_GEMINI_API_KEY"):
        provider._get_client()


@pytest.mark.asyncio
async def test_cloud_stt_transcribe(sample_wav):
    settings = STTSettings(mode=ProviderMode.CLOUD, gemini_api_key="test-key")
    provider = GeminiSTTProvider(settings)
    provider._client = MagicMock()  # skip _get_client

    with patch.object(
        GeminiSTTProvider, "_call_gemini", return_value=("  Привіт світ  ", None)
    ):
        result = await provider.transcribe(sample_wav, language="uk")

    assert result.text == "Привіт світ"


@pytest.mark.asyncio
async def test_cloud_stt_tokens_used(sample_wav):
    settings = STTSettings(mode=ProviderMode.CLOUD, gemini_api_key="test-key")
    provider = GeminiSTTProvider(settings)
    provider._client = MagicMock()

    with patch.object(GeminiSTTProvider, "_call_gemini", return_value=("Привіт світ", 1500)):
        result = await provider.transcribe(sample_wav, language="uk")

    assert result.tokens_used == 1500


@pytest.mark.asyncio
async def test_cloud_stt_empty_response(sample_wav):
    settings = STTSettings(mode=ProviderMode.CLOUD, gemini_api_key="test-key")
    provider = GeminiSTTProvider(settings)
    provider._client = MagicMock()

    with patch.object(GeminiSTTProvider, "_call_gemini", return_value=(None, None)):
        result = await provider.transcribe(sample_wav)

    assert result.text == ""


# --- Local STT ---


def test_local_stt_model_name():
    settings = STTSettings(mode=ProviderMode.LOCAL, whisper_model_size="large-v3")
    provider = LocalSTTProvider(settings)
    assert provider.model_name == "whisper/large-v3"


def test_local_stt_device_detection_no_torch():
    with patch.dict("sys.modules", {"torch": None}):
        assert LocalSTTProvider._detect_device() == "cpu"


@pytest.mark.asyncio
async def test_local_stt_transcribe(sample_wav):
    settings = STTSettings(mode=ProviderMode.LOCAL)
    provider = LocalSTTProvider(settings)

    mock_segment1 = MagicMock()
    mock_segment1.text = " Привіт "
    mock_segment2 = MagicMock()
    mock_segment2.text = " світ "

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], MagicMock())
    provider._model = mock_model

    result = await provider.transcribe(sample_wav, language="uk")

    assert result.text == "Привіт світ"
    mock_model.transcribe.assert_called_once()


def test_local_stt_last_load_error_starts_none():
    settings = STTSettings(mode=ProviderMode.LOCAL)
    provider = LocalSTTProvider(settings)
    assert provider.last_load_error is None
