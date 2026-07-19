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
    with pytest.raises(RuntimeError, match="missing"):
        provider._get_client()


@pytest.mark.asyncio
async def test_transcribe_returns_stripped_text(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()  # skip real SDK init

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value=("  привіт  ", None, None)):
        result = await provider.transcribe(_wav(tmp_path), language="uk")

    assert result.text == "привіт"


@pytest.mark.asyncio
async def test_transcribe_ignores_unknown_kwargs(tmp_path):
    """style=ai_prompt must not crash Groq — it just ignores it."""
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value=("ok", None, None)):
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
        provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None)


def test_other_errors_bubble_up_unchanged(tmp_path):
    """Non-429 SDK errors must propagate as-is so callers see the root cause."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = ValueError("invalid audio")

    with pytest.raises(ValueError, match="invalid audio"):
        provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None)


def test_cleanup_resets_client():
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is None


# --- STT quality wins: initial_prompt threading ---


@pytest.mark.asyncio
async def test_groq_threads_initial_prompt_when_set(tmp_path):
    """Non-empty glossary lands in the SDK call as `prompt=...`."""
    provider = GroqWhisperSTTProvider(_settings(initial_prompt="Tauri FastAPI"))
    provider._client = MagicMock()
    captured: dict = {}

    def _spy(client, model, audio_path, language, prompt):
        captured["prompt"] = prompt
        return "ok", None, None

    with patch.object(GroqWhisperSTTProvider, "_call_groq", side_effect=_spy):
        await provider.transcribe(_wav(tmp_path), language="uk")

    assert captured["prompt"] == "Tauri FastAPI"


@pytest.mark.asyncio
async def test_groq_omits_empty_prompt_to_avoid_400(tmp_path):
    """Empty/whitespace glossary must NOT become `prompt=""` — Groq rejects that with 400."""
    provider = GroqWhisperSTTProvider(_settings(initial_prompt="   "))
    provider._client = MagicMock()
    captured: dict = {}

    def _spy(client, model, audio_path, language, prompt):
        captured["prompt"] = prompt
        return "ok", None, None

    with patch.object(GroqWhisperSTTProvider, "_call_groq", side_effect=_spy):
        await provider.transcribe(_wav(tmp_path), language="uk")

    assert captured["prompt"] is None


def test_groq_sdk_payload_skips_prompt_key_when_none(tmp_path):
    """The Groq SDK call must NOT include a `prompt` key at all when none is set."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None)

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert "prompt" not in call_kwargs


def test_groq_sdk_payload_includes_prompt_when_set(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", "glossary text")

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["prompt"] == "glossary text"


# --- STT auto-detect (spec 019) ---


def test_groq_sdk_payload_includes_language_when_explicit_code(tmp_path):
    """Regression: an explicit BCP-47 code must still reach the Groq SDK as-is."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None)

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["language"] == "uk"


def test_groq_sdk_payload_omits_language_key_for_auto(tmp_path):
    """language="auto" must NOT be forwarded to the Groq SDK at all — omission
    is the documented auto-detect path (mirrors the SDK's own `Omit` default),
    never the literal string "auto" (unverified against Groq's closed-source
    server)."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "auto", None)

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert "language" not in call_kwargs


# --- detected_language / verbose_json escalation (spec 029, AC 19) -----------


def test_call_groq_uses_text_format_for_explicit_language(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    provider._call_groq(client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None)

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["response_format"] == "text"


def test_call_groq_escalates_to_verbose_json_for_auto(tmp_path):
    """AC-19: response_format escalates to verbose_json only when
    language == "auto" -- the explicit-language path keeps "text" unchanged."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text="hello", language="en")

    text, detected, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "auto", None
    )

    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["response_format"] == "verbose_json"
    assert text == "hello"
    assert detected == "en"


def test_call_groq_bare_string_response_has_no_detected_language(tmp_path):
    """Regression for the SDK-returns-a-bare-string case already handled at
    groq_whisper.py:112-115 -- response_format="text" returns a plain
    string with no metadata whatsoever."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "плейн текст"

    text, detected, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None
    )

    assert text == "плейн текст"
    assert detected is None


@pytest.mark.asyncio
async def test_transcribe_populates_normalized_detected_language(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value=("hello", "EN", None)):
        result = await provider.transcribe(_wav(tmp_path), language="auto")

    assert result.detected_language == "en"


@pytest.mark.asyncio
async def test_transcribe_detected_language_none_for_explicit_language(tmp_path):
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()

    with patch.object(GroqWhisperSTTProvider, "_call_groq", return_value=("привіт", None, None)):
        result = await provider.transcribe(_wav(tmp_path), language="uk")

    assert result.detected_language is None


# --- Spec 033 / AC 18: no_speech_prob off the verbose_json branch ---------


def test_call_groq_reads_min_no_speech_prob_from_object_segments(tmp_path):
    """AC-18: the SDK returns attribute-objects on some versions -- the min
    across segments must reach the caller."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()

    class _Seg:
        def __init__(self, p):
            self.no_speech_prob = p

    client.audio.transcriptions.create.return_value = MagicMock(
        text="hello", language="en", segments=[_Seg(0.77), _Seg(0.12)]
    )

    text, detected, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "auto", None
    )

    assert no_speech == 0.12
    assert text == "hello"
    assert detected == "en"


def test_call_groq_reads_min_no_speech_prob_from_dict_segments(tmp_path):
    """AC-18: ...and dicts on others. Both shapes, one reader."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(
        text="hello",
        language="en",
        segments=[{"no_speech_prob": 0.9}, {"no_speech_prob": 0.35}],
    )

    _, _, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "auto", None
    )

    assert no_speech == 0.35


def test_call_groq_text_path_reports_no_no_speech_prob(tmp_path):
    """AC-18: the explicit-language hot path uses response_format="text" --
    a bare string with no metadata whatsoever. Escalating it to
    verbose_json would re-open the blast radius ADR 016 bounded."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "плейн текст"

    text, detected, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None
    )

    assert text == "плейн текст"
    assert detected is None
    assert no_speech is None


def test_call_groq_missing_segments_fails_open(tmp_path):
    """AC-18: Groq's inference server is closed-source and its payload shape
    is not contractual -- a response without usable segments must yield None
    (keep the transcription), never raise."""
    provider = GroqWhisperSTTProvider(_settings())
    client = MagicMock(spec=[])
    response = MagicMock(spec=["text", "language"])
    response.text = "hello"
    response.language = "en"
    client.audio = MagicMock()
    client.audio.transcriptions = MagicMock()
    client.audio.transcriptions.create = MagicMock(return_value=response)

    _, _, no_speech = provider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "auto", None
    )

    assert no_speech is None


@pytest.mark.asyncio
async def test_transcribe_threads_no_speech_prob_onto_the_contract(tmp_path):
    """AC-18: the value _call_groq returns actually lands on
    TranscriptionResult, where the pipeline's layer-3 gate reads it."""
    provider = GroqWhisperSTTProvider(_settings())
    provider._client = MagicMock()

    with patch.object(
        GroqWhisperSTTProvider, "_call_groq", return_value=("привіт", "uk", 0.88)
    ):
        result = await provider.transcribe(_wav(tmp_path), language="auto")

    assert result.no_speech_prob == 0.88
