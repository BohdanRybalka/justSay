import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi import BackgroundTasks

from app.core import history
from app.core.config import settings
from app.core.types import ProviderMode
from app.pipeline.service import process_audio
from app.stt.base import TranscriptionResult


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path, monkeypatch):
    """spec 017: process_audio now also schedules
    ``vector_store.run_background_indexer`` in the background, which — once
    a test actually invokes the scheduled ``BackgroundTasks`` queue (see
    ``test_pipeline_survives_embedding_provider_outage``) — queries the real
    ``history`` module's connection. Without this isolation those queries
    would hit whatever real ``history.db`` / ``_output_dir`` happens to be
    live on the machine running the suite, matching the ``isolated_storage``
    convention already used by test_history_sqlite.py / test_vector_store.py
    / test_words.py."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


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


def _make_stt_mock(text: str = "hello world", tokens: int | None = None):
    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text=text, tokens_used=tokens))
    stt.model_name = "mock/provider"
    return stt


@pytest.mark.asyncio
async def test_pipeline_returns_stt_text_verbatim(
    sample_wav, cloud_mode, _isolate_side_effects
):
    copy_mock, save_mock = _isolate_side_effects
    stt = _make_stt_mock("Привіт світ")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.text == "Привіт світ"
    assert result.copied_to_clipboard is True
    copy_mock.assert_called_once_with("Привіт світ")
    saved_kwargs = save_mock.call_args.kwargs
    assert saved_kwargs["text"] == "Привіт світ"
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

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.copied_to_clipboard is False
    copy_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_clipboard_failure_is_graceful(
    sample_wav, cloud_mode, _isolate_side_effects
):
    copy_mock, save_mock = _isolate_side_effects
    copy_mock.side_effect = RuntimeError("no clipboard")
    stt = _make_stt_mock("text")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.text == "text"
    assert result.copied_to_clipboard is False
    # Even when clipboard fails, the entry must still be saved.
    assert save_mock.call_count == 1
    assert save_mock.call_args.kwargs["text"] == "text"


@pytest.mark.asyncio
async def test_pipeline_passes_style_to_provider(
    sample_wav, cloud_mode, _isolate_side_effects
):
    stt = _make_stt_mock("structured output")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)) as routed:
        await process_audio(sample_wav, language="uk", style="ai_prompt")

    routed.assert_called_once()
    _, kwargs = routed.call_args
    assert kwargs["style"] == "ai_prompt"

    stt.transcribe.assert_awaited_once()
    call_kwargs = stt.transcribe.await_args.kwargs
    assert call_kwargs.get("style") == "ai_prompt"


@pytest.mark.asyncio
async def test_pipeline_records_ai_prompt_style_in_history(
    sample_wav, cloud_mode, _isolate_side_effects
):
    _, save_mock = _isolate_side_effects
    stt = _make_stt_mock("ok")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        await process_audio(sample_wav, language="uk", style="ai_prompt")

    assert save_mock.call_args.kwargs["style"] == "ai_prompt"


@pytest.mark.asyncio
async def test_pipeline_forwards_tokens_used_to_history(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """tokens_used from STT result must reach save_entry unchanged."""
    _, save_mock = _isolate_side_effects
    stt = _make_stt_mock("hello", tokens=1500)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        await process_audio(sample_wav, language="uk", style="normal")

    assert save_mock.call_args.kwargs["tokens_used"] == 1500


@pytest.mark.asyncio
async def test_pipeline_respects_explicit_audio_duration(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """When caller provides audio_duration, pipeline must not re-detect it."""
    stt = _make_stt_mock("ok")

    with patch("app.pipeline.service.detect_duration") as detect, patch(
        "app.pipeline.service.get_routed_provider", return_value=(stt, None)
    ) as routed:
        await process_audio(sample_wav, audio_duration=12.5, style="normal")

    detect.assert_not_called()
    assert routed.call_args.kwargs["audio_duration"] == 12.5


@pytest.mark.asyncio
async def test_pipeline_passes_file_extension_to_routing(
    sample_wav, cloud_mode, _isolate_side_effects
):
    stt = _make_stt_mock("ok")
    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)) as routed:
        await process_audio(sample_wav, style="normal")
    assert routed.call_args.kwargs["file_extension"] == ".wav"


@pytest.mark.asyncio
async def test_pipeline_propagates_stt_failure(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """STT exception must bubble up; no history entry created."""
    _, save_mock = _isolate_side_effects
    stt = MagicMock()
    stt.transcribe = AsyncMock(side_effect=RuntimeError("groq down"))
    stt.model_name = "mock/provider"

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        with pytest.raises(RuntimeError, match="groq down"):
            await process_audio(sample_wav, style="normal")

    save_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_concurrent_invocations_save_independently(
    cloud_mode, _isolate_side_effects, tmp_path
):
    """Five parallel pipeline runs must each emit one save_entry."""
    _, save_mock = _isolate_side_effects

    paths = []
    for i in range(5):
        audio = np.random.uniform(-0.05, 0.05, 16000).astype(np.float32)
        p = tmp_path / f"sample-{i}.wav"
        sf.write(str(p), audio, 16000)
        paths.append(p)

    def make_stt(idx: int):
        m = MagicMock()
        m.transcribe = AsyncMock(return_value=TranscriptionResult(text=f"text-{idx}", tokens_used=None))
        m.model_name = "mock/provider"
        return m

    async def one(idx: int):
        with patch("app.pipeline.service.get_routed_provider", return_value=(make_stt(idx), None)):
            r = await process_audio(paths[idx], style="normal")
        return r

    results = await asyncio.gather(*[one(i) for i in range(5)])
    texts = sorted(r.text for r in results)
    assert texts == [f"text-{i}" for i in range(5)]
    assert save_mock.call_count == 5


# --- Latency isolation (spec 003) -------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_schedules_embedding_via_background_tasks_not_awaited(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """The call site MUST be background_tasks.add_task(
    vector_store.embed_entry_background, entry.id, text), never a direct
    `await embed_entry_background(...)` inside process_audio — this is the
    structural guarantee that embedding latency cannot land inside the
    request/response cycle. spec 017: a second task,
    `vector_store.run_background_indexer` (no args), must also be
    scheduled alongside it, for every dictation that produces text."""
    _, save_mock = _isolate_side_effects
    fake_entry = MagicMock()
    fake_entry.id = "entry-123"
    save_mock.return_value = fake_entry
    stt = _make_stt_mock("hello world")

    bt = BackgroundTasks()
    with (
        patch.object(bt, "add_task") as add_task_mock,
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, style="normal", background_tasks=bt)

    assert result.text == "hello world"
    assert add_task_mock.call_count == 2

    from app.core import vector_store

    embed_call, indexer_call = add_task_mock.call_args_list

    assert embed_call.args[0] is vector_store.embed_entry_background
    assert embed_call.args[1] == "entry-123"
    assert embed_call.args[2] == "hello world"

    assert indexer_call.args[0] is vector_store.run_background_indexer
    assert indexer_call.args[1:] == ()


@pytest.mark.asyncio
async def test_pipeline_omits_background_task_when_none_provided(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """background_tasks=None (the default) must not attempt to schedule
    anything — covers callers that don't pass BackgroundTasks at all."""
    stt = _make_stt_mock("hello world")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, style="normal")  # no background_tasks

    assert result.text == "hello world"  # must not raise


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_pipeline_survives_embedding_provider_outage(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """A failure inside the scheduled background embed task must not
    affect process_audio's return value or the saved HistoryEntry —
    dictation succeeds regardless of embedding failure. process_audio
    returns BEFORE the background task ever runs; this test then runs the
    queued task the way Starlette's response middleware would (after the
    response is sent) and asserts it does not raise and does not retroactively
    change anything process_audio already returned.

    Marked @pytest.mark.background_indexer (spec 017 review triage, closes
    RED #1): this test invokes the scheduled BackgroundTasks queue for real
    via `await bt()`, which also runs the real `run_background_indexer`
    task scheduled alongside `embed_entry_background` — without this
    marker, the new conftest.py autouse fixture would silently replace it
    with a no-op, and this test would stop covering what it invokes. The
    `_isolated_history` fixture (this file's own module-level autouse
    fixture, added as Deviation 2) still protects the real DB access this
    causes: `save_entry` is mocked via `_isolate_side_effects` so no entry
    actually lands in the isolated `entries` table, and
    `run_background_indexer` finds an empty backlog and returns immediately."""
    _, save_mock = _isolate_side_effects
    fake_entry = MagicMock()
    fake_entry.id = "entry-456"
    save_mock.return_value = fake_entry
    stt = _make_stt_mock("hello world")

    bt = BackgroundTasks()
    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, style="normal", background_tasks=bt)

    # process_audio already returned successfully; the background task has
    # not run yet at this point.
    assert result.text == "hello world"
    assert result.copied_to_clipboard is True
    assert save_mock.call_count == 1

    with (
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(side_effect=RuntimeError("embedding provider outage")),
        ),
        patch("app.core.history._vec_available", True),
    ):
        await bt()  # must not raise — embed_entry_background swallows it

    # process_audio's earlier return value / saved entry are unaffected.
    assert result.text == "hello world"
    assert save_mock.call_args.kwargs["text"] == "hello world"
