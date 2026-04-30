from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.core.config import settings
from app.core.types import ProviderMode
from app.pipeline.service import process_audio
from app.stt.base import TranscriptionResult


@pytest.fixture
def sample_wav(tmp_path) -> Path:
    audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)  # 1s mono 16kHz
    path = tmp_path / "sample.wav"
    sf.write(str(path), audio, 16000)
    return path


@pytest.fixture(autouse=True)
def _isolate_side_effects():
    """Clipboard writes and history persistence are side-effects we mock out."""
    with patch("app.pipeline.service.pyperclip.copy") as copy_mock, patch(
        "app.pipeline.service.save_entry"
    ) as save_mock:
        yield copy_mock, save_mock


@pytest.fixture
def cloud_mode():
    original_stt = settings.stt.mode
    original_llm = settings.llm.mode
    settings.stt.mode = ProviderMode.CLOUD
    settings.llm.mode = ProviderMode.CLOUD
    yield
    settings.stt.mode = original_stt
    settings.llm.mode = original_llm


def _make_stt_mock(text: str = "hello world"):
    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text=text, tokens_used=None))
    stt.model_name = "mock/provider"
    return stt


@pytest.mark.asyncio
async def test_pipeline_returns_stt_text_verbatim(
    sample_wav, cloud_mode, _isolate_side_effects
):
    copy_mock, save_mock = _isolate_side_effects
    stt = _make_stt_mock("Привіт світ")

    with patch("app.pipeline.service.get_routed_provider", return_value=stt):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.raw_text == "Привіт світ"
    assert result.cleaned_text == "Привіт світ"
    assert result.copied_to_clipboard is True
    copy_mock.assert_called_once_with("Привіт світ")
    saved_kwargs = save_mock.call_args.kwargs
    assert saved_kwargs["raw_text"] == "Привіт світ"
    assert saved_kwargs["cleaned_text"] == "Привіт світ"
    assert saved_kwargs["language"] == "uk"
    assert saved_kwargs["style"] == "normal"
    assert saved_kwargs["model_name"] == "mock/provider"
    assert saved_kwargs["tokens_used"] is None
    assert saved_kwargs["word_count"] == 2
    assert isinstance(saved_kwargs["duration_ms"], int) and saved_kwargs["duration_ms"] >= 0
    assert isinstance(saved_kwargs["audio_duration_seconds"], float)
    assert saved_kwargs["audio_duration_seconds"] > 0


@pytest.mark.asyncio
async def test_pipeline_does_not_copy_empty_text(
    sample_wav, cloud_mode, _isolate_side_effects
):
    copy_mock, _ = _isolate_side_effects
    stt = _make_stt_mock("")

    with patch("app.pipeline.service.get_routed_provider", return_value=stt):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.copied_to_clipboard is False
    copy_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_clipboard_failure_is_graceful(
    sample_wav, cloud_mode, _isolate_side_effects
):
    copy_mock, _ = _isolate_side_effects
    copy_mock.side_effect = RuntimeError("no clipboard")
    stt = _make_stt_mock("text")

    with patch("app.pipeline.service.get_routed_provider", return_value=stt):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.cleaned_text == "text"
    assert result.copied_to_clipboard is False


@pytest.mark.asyncio
async def test_pipeline_passes_style_to_provider(
    sample_wav, cloud_mode, _isolate_side_effects
):
    stt = _make_stt_mock("structured output")

    with patch("app.pipeline.service.get_routed_provider", return_value=stt) as routed:
        await process_audio(sample_wav, language="uk", style="ai_prompt")

    # Provider selected with the right context...
    routed.assert_called_once()
    _, kwargs = routed.call_args
    assert kwargs["style"] == "ai_prompt"

    # ...and style is forwarded to transcribe(**kwargs).
    stt.transcribe.assert_awaited_once()
    call_kwargs = stt.transcribe.await_args.kwargs
    assert call_kwargs.get("style") == "ai_prompt"


@pytest.mark.asyncio
async def test_pipeline_forwards_tokens_used_to_history(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """tokens_used from STT result must reach save_entry unchanged."""
    _, save_mock = _isolate_side_effects
    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="hello", tokens_used=1500))
    stt.model_name = "mock/provider"

    with patch("app.pipeline.service.get_routed_provider", return_value=stt):
        await process_audio(sample_wav, language="uk", style="normal")

    assert save_mock.call_args.kwargs["tokens_used"] == 1500


@pytest.mark.asyncio
async def test_pipeline_respects_explicit_audio_duration(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """When caller provides audio_duration, pipeline must not re-detect it."""
    stt = _make_stt_mock("ok")

    with patch("app.pipeline.service.detect_duration") as detect, patch(
        "app.pipeline.service.get_routed_provider", return_value=stt
    ) as routed:
        await process_audio(sample_wav, audio_duration=12.5, style="normal")

    detect.assert_not_called()
    assert routed.call_args.kwargs["audio_duration"] == 12.5
