from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.stt import (
    GEMINI_SUPPORTED_FORMATS,
    GROQ_SUPPORTED_FORMATS,
    clear_cache,
    get_routed_provider,
)
from app.stt.cloud import GeminiSTTProvider
from app.stt.config import STTSettings
from app.stt.groq_whisper import GroqWhisperSTTProvider
from app.stt.local import LocalSTTProvider


@pytest.fixture(autouse=True)
def _clear_stt_cache():
    clear_cache()
    yield
    clear_cache()


def _cloud_settings(**overrides) -> STTSettings:
    defaults = dict(
        mode=ProviderMode.CLOUD,
        gemini_api_key="g",
        groq_api_key="q",
        cloud_routing_threshold=30.0,
    )
    defaults.update(overrides)
    return STTSettings(**defaults)


# --- Mode-level dispatch ---


def test_local_mode_always_returns_local():
    s = STTSettings(mode=ProviderMode.LOCAL)
    p = get_routed_provider(s, audio_duration=5.0, style="normal")
    assert isinstance(p, LocalSTTProvider)


def test_local_mode_ignores_style_and_duration():
    s = STTSettings(mode=ProviderMode.LOCAL)
    p = get_routed_provider(s, audio_duration=600.0, style="ai_prompt")
    assert isinstance(p, LocalSTTProvider)


# --- Cloud routing by duration ---


def test_short_normal_goes_to_groq():
    s = _cloud_settings()
    p = get_routed_provider(s, audio_duration=10.0, style="normal")
    assert isinstance(p, GroqWhisperSTTProvider)


def test_threshold_boundary_exact_goes_to_groq():
    s = _cloud_settings(cloud_routing_threshold=30.0)
    p = get_routed_provider(s, audio_duration=30.0, style="normal")
    assert isinstance(p, GroqWhisperSTTProvider)


def test_long_normal_goes_to_gemini():
    s = _cloud_settings()
    p = get_routed_provider(s, audio_duration=60.0, style="normal")
    assert isinstance(p, GeminiSTTProvider)


def test_unknown_duration_falls_back_to_gemini():
    s = _cloud_settings()
    p = get_routed_provider(s, audio_duration=None, style="normal")
    assert isinstance(p, GeminiSTTProvider)


# --- Cloud routing by style ---


def test_ai_prompt_always_goes_to_gemini_regardless_of_duration():
    s = _cloud_settings()
    short = get_routed_provider(s, audio_duration=5.0, style="ai_prompt")
    long = get_routed_provider(s, audio_duration=120.0, style="ai_prompt")
    assert isinstance(short, GeminiSTTProvider)
    assert isinstance(long, GeminiSTTProvider)


# --- Cloud routing by file extension ---


def test_webm_short_normal_falls_back_to_gemini():
    """Groq can't handle .webm — router must degrade to Gemini."""
    s = _cloud_settings()
    p = get_routed_provider(s, audio_duration=5.0, style="normal", file_extension=".webm")
    assert isinstance(p, GeminiSTTProvider)


def test_wav_short_normal_uses_groq():
    s = _cloud_settings()
    p = get_routed_provider(s, audio_duration=5.0, style="normal", file_extension=".wav")
    assert isinstance(p, GroqWhisperSTTProvider)


def test_format_supports_sets_are_consistent():
    """Sanity check on the supported-format tables."""
    assert ".wav" in GROQ_SUPPORTED_FORMATS
    assert ".webm" not in GROQ_SUPPORTED_FORMATS
    assert ".webm" in GEMINI_SUPPORTED_FORMATS


# --- Cache behaviour ---


def test_same_provider_is_cached_across_calls():
    s = _cloud_settings()
    p1 = get_routed_provider(s, audio_duration=5.0, style="normal")
    p2 = get_routed_provider(s, audio_duration=10.0, style="normal")
    assert p1 is p2  # both Groq


def test_different_providers_coexist_in_cache():
    s = _cloud_settings()
    groq = get_routed_provider(s, audio_duration=5.0, style="normal")
    gemini = get_routed_provider(s, audio_duration=100.0, style="normal")
    assert isinstance(groq, GroqWhisperSTTProvider)
    assert isinstance(gemini, GeminiSTTProvider)
    assert groq is not gemini


def test_clear_cache_triggers_cleanup_on_all():
    s = _cloud_settings()
    groq = get_routed_provider(s, audio_duration=5.0, style="normal")
    gemini = get_routed_provider(s, audio_duration=100.0, style="normal")

    with patch.object(groq, "cleanup") as gc_mock, patch.object(gemini, "cleanup") as gm_mock:
        clear_cache()
        gc_mock.assert_called_once()
        gm_mock.assert_called_once()


# --- Config validator ---


def test_cloud_routing_threshold_must_be_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        STTSettings(cloud_routing_threshold=0)

    with pytest.raises(ValueError, match="must be > 0"):
        STTSettings(cloud_routing_threshold=-5)


# --- detect_duration ---


def test_detect_duration_returns_none_for_missing_file(tmp_path):
    from app.pipeline.utils import detect_duration

    assert detect_duration(tmp_path / "no-such-file.wav") is None


def test_detect_duration_reads_real_wav(tmp_path):
    import numpy as np
    import soundfile as sf
    from app.pipeline.utils import detect_duration

    path = tmp_path / "two-seconds.wav"
    samples = np.zeros(32000, dtype=np.float32)  # 2s @ 16kHz
    sf.write(str(path), samples, 16000)

    duration = detect_duration(path)
    assert duration is not None
    assert 1.9 < duration < 2.1
