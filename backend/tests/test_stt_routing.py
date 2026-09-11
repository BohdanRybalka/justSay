from unittest.mock import patch

import pytest

from app.core.audio_formats import ALLOWED_AUDIO_EXTENSIONS
from app.core.types import ProviderMode
from app.stt import (
    GROQ_SUPPORTED_FORMATS,
    _providers,
    clear_cache,
    get_routed_provider,
)
from app.stt.cloud import GeminiSTTProvider
from app.stt.config import STTSettings
from app.stt.groq_whisper import GroqWhisperSTTProvider
from app.stt.local import LocalSTTProvider
from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider


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


def test_local_mode_always_returns_local():
    """Proves only that LOCAL does not reach a cloud provider — see the marked
    test at the end of this file for the platform-routing pin (JS-97)."""
    s = STTSettings(mode=ProviderMode.LOCAL)
    p, fallback = get_routed_provider(s, audio_duration=5.0)
    assert isinstance(p, LocalSTTProvider)
    assert fallback is None


def test_local_mode_reaches_no_cloud_provider_on_any_input():
    """The zero-leak guarantee at the Local/Cloud decision point itself.

    Sweeps every input ``get_routed_provider`` still accepts — 3 engine pins x
    4 durations x 14 file extensions — and makes two different claims about
    each: the returned provider declares ``is_local``, so the pipeline's
    ``transcribe`` cannot be a cloud call; and the module cache holds no cloud
    provider afterwards, so no cloud client was ever constructed and no API key
    was ever read.

    It is a statement about this function, not about the process: embedding
    generation, settings sync and the update check are other paths.
    """
    durations = [None, 5.0, 30.0, 600.0]
    extensions = [None, *sorted(ALLOWED_AUDIO_EXTENSIONS)]
    combinations = 0
    for engine in ("auto", "groq", "gemini"):
        for duration in durations:
            for extension in extensions:
                clear_cache()
                s = STTSettings(
                    mode=ProviderMode.LOCAL,
                    engine=engine,
                    gemini_api_key="g",
                    groq_api_key="q",
                    cloud_routing_threshold=30.0,
                )
                provider, fallback = get_routed_provider(
                    s, audio_duration=duration, file_extension=extension
                )
                combinations += 1
                assert provider.is_local is True, (
                    f"engine={engine} duration={duration} ext={extension} routed to "
                    f"{type(provider).__name__}, which is not a local provider"
                )
                assert fallback is None
                cloud_cached = [
                    cls.__name__
                    for cls in _providers
                    if cls in (GeminiSTTProvider, GroqWhisperSTTProvider)
                ]
                assert not cloud_cached, (
                    f"engine={engine} duration={duration} ext={extension} constructed "
                    f"{cloud_cached} in Local mode"
                )
    assert combinations == 168


def test_local_mode_ignores_duration():
    s = STTSettings(mode=ProviderMode.LOCAL)
    p, _ = get_routed_provider(s, audio_duration=600.0)
    assert isinstance(p, LocalSTTProvider)


@pytest.mark.no_factory_stub
@pytest.mark.parametrize("duration", [5.0, 600.0])
def test_local_mode_routes_to_this_platforms_local_provider(monkeypatch, duration):
    """Neither unmarked sibling can say this: the autouse
    `_force_faster_whisper_for_local` fixture pins the class they assert."""
    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: WhisperCppServerSTTProvider,
    )
    clear_cache()
    s = STTSettings(mode=ProviderMode.LOCAL)
    p, fallback = get_routed_provider(s, audio_duration=duration)
    assert type(p) is WhisperCppServerSTTProvider
    assert fallback is None


def test_short_normal_goes_to_groq():
    s = _cloud_settings()
    p, fallback = get_routed_provider(s, audio_duration=10.0)
    assert isinstance(p, GroqWhisperSTTProvider)
    assert fallback is None


def test_threshold_boundary_exact_goes_to_groq():
    s = _cloud_settings(cloud_routing_threshold=30.0)
    p, _ = get_routed_provider(s, audio_duration=30.0)
    assert isinstance(p, GroqWhisperSTTProvider)


def test_long_normal_goes_to_gemini():
    s = _cloud_settings()
    p, _ = get_routed_provider(s, audio_duration=60.0)
    assert isinstance(p, GeminiSTTProvider)


def test_unknown_duration_falls_back_to_gemini():
    s = _cloud_settings()
    p, _ = get_routed_provider(s, audio_duration=None)
    assert isinstance(p, GeminiSTTProvider)


def test_webm_short_normal_falls_back_to_gemini():
    """Groq can't handle .webm — router must degrade to Gemini."""
    s = _cloud_settings()
    p, _ = get_routed_provider(s, audio_duration=5.0, file_extension=".webm")
    assert isinstance(p, GeminiSTTProvider)


def test_wav_short_normal_uses_groq():
    s = _cloud_settings()
    p, _ = get_routed_provider(s, audio_duration=5.0, file_extension=".wav")
    assert isinstance(p, GroqWhisperSTTProvider)


def test_groq_advertises_no_extension_the_upload_allowlist_rejects():
    """Groq's table is the routing input, and it can only narrow the allowlist.

    A provider advertising an extension the upload validator rejects is a
    routing decision that can never be reached: ``upload_validation`` refuses
    the file before ``get_routed_provider`` is consulted.

    Mutation-checked: adding ".mkv" to GROQ_SUPPORTED_FORMATS fails this test
    naming .mkv as an extension the upload allowlist does not accept.
    """
    assert ".wav" in GROQ_SUPPORTED_FORMATS
    assert ".webm" not in GROQ_SUPPORTED_FORMATS
    assert ".webm" in ALLOWED_AUDIO_EXTENSIONS
    unaccepted = GROQ_SUPPORTED_FORMATS - ALLOWED_AUDIO_EXTENSIONS
    assert not unaccepted, (
        "GROQ_SUPPORTED_FORMATS advertises extensions the upload allowlist does not "
        f"accept, so routing to Groq could never be reached for them: {sorted(unaccepted)}"
    )


def test_same_provider_is_cached_across_calls():
    s = _cloud_settings()
    p1, _ = get_routed_provider(s, audio_duration=5.0)
    p2, _ = get_routed_provider(s, audio_duration=10.0)
    assert p1 is p2


def test_different_providers_coexist_in_cache():
    s = _cloud_settings()
    groq, _ = get_routed_provider(s, audio_duration=5.0)
    gemini, _ = get_routed_provider(s, audio_duration=100.0)
    assert isinstance(groq, GroqWhisperSTTProvider)
    assert isinstance(gemini, GeminiSTTProvider)
    assert groq is not gemini


def test_clear_cache_triggers_cleanup_on_all():
    s = _cloud_settings()
    groq, _ = get_routed_provider(s, audio_duration=5.0)
    gemini, _ = get_routed_provider(s, audio_duration=100.0)

    with patch.object(groq, "cleanup") as gc_mock, patch.object(gemini, "cleanup") as gm_mock:
        clear_cache()
        gc_mock.assert_called_once()
        gm_mock.assert_called_once()


def test_engine_pin_groq_overrides_long_audio():
    s = _cloud_settings(engine="groq")
    p, fallback = get_routed_provider(s, audio_duration=600.0)
    assert isinstance(p, GroqWhisperSTTProvider)
    assert fallback is None


def test_engine_pin_groq_falls_back_for_unsupported_format():
    s = _cloud_settings(engine="groq")
    p, fallback = get_routed_provider(
        s, audio_duration=5.0, file_extension=".webm"
    )
    assert isinstance(p, GeminiSTTProvider)
    assert fallback and ".webm" in fallback


def test_engine_pin_gemini_overrides_short_audio():
    s = _cloud_settings(engine="gemini")
    p, fallback = get_routed_provider(s, audio_duration=2.0)
    assert isinstance(p, GeminiSTTProvider)
    assert fallback is None


def test_cloud_routing_threshold_must_be_positive():
    with pytest.raises(ValueError):
        STTSettings(cloud_routing_threshold=0)

    with pytest.raises(ValueError):
        STTSettings(cloud_routing_threshold=-5)


def test_detect_duration_returns_none_for_missing_file(tmp_path):
    from app.pipeline.utils import detect_duration

    assert detect_duration(tmp_path / "no-such-file.wav") is None


def test_detect_duration_reads_real_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from app.pipeline.utils import detect_duration

    path = tmp_path / "two-seconds.wav"
    samples = np.zeros(32000, dtype=np.float32)
    sf.write(str(path), samples, 16000)

    duration = detect_duration(path)
    assert duration is not None
    assert 1.9 < duration < 2.1
