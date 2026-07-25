import asyncio
import time
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
    audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
    path = tmp_path / "sample.wav"
    sf.write(str(path), audio, 16000)
    return path


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """1s of true digital silence -- well past the silence guard's default
    thresholds (spec 029)."""
    audio = np.zeros(16000, dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, 16000)
    return path


@pytest.fixture(autouse=True)
def _vad_abstains(request):
    """Pin the spec-033 neural VAD layer to "abstain" (``None``) by default.

    Without this the suite is non-deterministic across machines: on a
    checkout where `backend/scripts/fetch_ten_vad.py` has been run, the real
    TEN VAD binary loads and correctly judges these synthetic
    uniform-noise/tone fixtures to be non-speech — discarding them and
    failing tests that were never about the VAD at all. ``None`` is the
    honest default here because it means exactly "no VAD verdict available",
    which is the shipped behaviour on every non-Windows platform and every
    checkout without the binary — i.e. these tests keep asserting the
    spec-029 energy-only path, bit-identically.

    Tests that ARE about the VAD opt out with ``@pytest.mark.no_vad_stub``
    and patch `app.pipeline.service.analyze_vad` themselves.
    """
    if "no_vad_stub" in request.keywords:
        yield None
        return
    with patch("app.pipeline.service.analyze_vad", return_value=None) as vad_mock:
        yield vad_mock


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
    stt.is_local = False
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
    stt.is_local = False

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
        m.transcribe = AsyncMock(
            return_value=TranscriptionResult(text=f"text-{idx}", tokens_used=None)
        )
        m.model_name = "mock/provider"
        m.is_local = False
        return m

    mocks = [make_stt(i) for i in range(5)]

    def _route(*args, **kwargs):
        return mocks.pop(0), None

    async def one(idx: int):
        return await process_audio(paths[idx], style="normal")

    with patch("app.pipeline.service.get_routed_provider", side_effect=_route):
        results = await asyncio.gather(*[one(i) for i in range(5)])

    texts = sorted(r.text for r in results)
    assert texts == [f"text-{i}" for i in range(5)]
    assert save_mock.call_count == 5



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
        result = await process_audio(sample_wav, style="normal")

    assert result.text == "hello world"


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
        await bt()

    assert result.text == "hello world"
    assert save_mock.call_args.kwargs["text"] == "hello world"




@pytest.mark.asyncio
async def test_pipeline_silence_guard_skips_stt_call(
    silent_wav, cloud_mode, _isolate_side_effects
):
    """AC-5: on a silent input, stt.transcribe is never called at all."""
    stt = _make_stt_mock("Дякую за перегляд")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(silent_wav, language="uk", style="normal")

    stt.transcribe.assert_not_called()
    assert result.text == ""
    assert result.copied_to_clipboard is False
    assert result.discarded_reason == "silence"


@pytest.mark.asyncio
async def test_pipeline_silence_guard_skips_clipboard_history_and_embeddings(
    silent_wav, cloud_mode, _isolate_side_effects
):
    """AC-6: no pyperclip.copy, no save_entry, no scheduled background
    embedding tasks -- all four asserted via _isolate_side_effects plus a
    real BackgroundTasks add_task spy."""
    copy_mock, save_mock = _isolate_side_effects
    stt = _make_stt_mock("Дякую за перегляд")

    bt = BackgroundTasks()
    with (
        patch.object(bt, "add_task") as add_task_mock,
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(
            silent_wav, language="uk", style="normal", background_tasks=bt
        )

    assert result.text == ""
    assert result.copied_to_clipboard is False
    assert result.discarded_reason == "silence"
    copy_mock.assert_not_called()
    save_mock.assert_not_called()
    add_task_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_silence_guard_does_not_raise(
    silent_wav, cloud_mode, _isolate_side_effects
):
    """AC-7: a discarded accidental hotkey press must be a silent no-op --
    the widget's error toast is wired to thrown exceptions."""
    stt = _make_stt_mock("Дякую за перегляд")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(silent_wav, style="normal")

    assert result.discarded_reason == "silence"


@pytest.mark.asyncio
async def test_pipeline_silence_guard_logs_warning_with_measurements(
    silent_wav, cloud_mode, _isolate_side_effects, caplog
):
    """AC-11: every silence discard logs at WARNING with the measured
    peak_dbfs and speech_frame_count, so a false positive from a low-gain
    mic is diagnosable from the log alone."""
    import logging

    stt = _make_stt_mock("Дякую за перегляд")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        with caplog.at_level(logging.WARNING, logger="app.pipeline.service"):
            await process_audio(silent_wav, style="normal")

    full_log = "\n".join(r.getMessage() for r in caplog.records)
    assert "peak=" in full_log
    assert "speech_frames=" in full_log


@pytest.mark.asyncio
async def test_pipeline_non_silent_audio_is_not_discarded(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """Regression guard: the existing sample_wav fixture's random noise
    (uniform -0.1..0.1) must clear the silence guard's thresholds -- this
    pins down that the guard doesn't start eating the rest of this test
    file's "real speech" fixtures."""
    stt = _make_stt_mock("hello world")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, style="normal")

    assert result.discarded_reason is None
    stt.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_silence_guard_does_not_block_event_loop(
    silent_wav, cloud_mode, _isolate_side_effects, monkeypatch
):
    """AC-32: analyze_silence runs via asyncio.to_thread -- a slow (real,
    time.sleep-based) analysis must not stall the event loop, mirroring the
    established pattern already used for local.py's _transcribe (Stage 3
    review YELLOW finding)."""
    import time

    from app.audio import analysis as analysis_module

    real_analyze_silence = analysis_module.analyze_silence

    def _slow_analyze_silence(*args, **kwargs):
        time.sleep(0.2)
        return real_analyze_silence(*args, **kwargs)

    monkeypatch.setattr("app.pipeline.service.analyze_silence", _slow_analyze_silence)

    stt = _make_stt_mock("should not be reached")

    done = asyncio.Event()
    ticks = {"n": 0}

    async def _ticker():
        while not done.is_set():
            await asyncio.sleep(0)
            ticks["n"] += 1

    async def _run():
        try:
            with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
                return await process_audio(silent_wav, style="normal")
        finally:
            done.set()

    results = await asyncio.gather(_run(), _ticker())
    result = results[0]

    assert ticks["n"] > 1, "event loop was blocked during analyze_silence()"
    assert result.discarded_reason == "silence"




@pytest.fixture
def local_mode():
    """Mirrors `cloud_mode` above but for Local -- restores the real STT
    provider cache afterwards since these tests route through the real
    LocalSTTProvider class (pinned by conftest.py's autouse
    `_force_faster_whisper_for_local` fixture), not a mocked provider."""
    from app.stt import clear_cache as clear_stt_cache

    original_stt = settings.stt.mode
    settings.stt.mode = ProviderMode.LOCAL
    clear_stt_cache()
    yield
    settings.stt.mode = original_stt
    clear_stt_cache()


@pytest.mark.asyncio
async def test_process_audio_skips_readiness_barrier_for_cloud_provider(
    sample_wav, cloud_mode, _isolate_side_effects, monkeypatch
):
    """AC 10: the barrier must not even be consulted for a cloud-routed
    request."""
    stt = _make_stt_mock("hello")

    def _boom(*args, **kwargs):
        raise AssertionError("await_local_ready must not be called for a cloud provider")

    monkeypatch.setattr("app.stt.local_setup.await_local_ready", _boom)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, style="normal")

    assert result.text == "hello"


@pytest.mark.asyncio
async def test_process_audio_awaits_shared_readiness_barrier_no_second_get_model(
    sample_wav, local_mode, _isolate_side_effects, monkeypatch
):
    """AC 11: two concurrent process_audio calls routed to the same
    not-yet-loaded local provider must not trigger a second _get_model() --
    the readiness barrier shares local_setup.ensure_local_ready()'s
    _prewarm_lock, asserted by call count, not by timing."""
    from app.stt.base import TranscriptionResult
    from app.stt.local import LocalSTTProvider

    _, save_mock = _isolate_side_effects
    call_count = {"n": 0}

    def _slow_get_model(self):
        call_count["n"] += 1
        time.sleep(0.05)
        self._model = object()

    monkeypatch.setattr(LocalSTTProvider, "_get_model", _slow_get_model)
    monkeypatch.setattr(
        LocalSTTProvider,
        "transcribe",
        AsyncMock(return_value=TranscriptionResult(text="ok", tokens_used=None)),
    )

    await asyncio.gather(
        process_audio(sample_wav, style="normal"),
        process_audio(sample_wav, style="normal"),
    )

    assert call_count["n"] == 1
    assert save_mock.call_count == 2


@pytest.mark.asyncio
async def test_process_audio_raises_clear_error_when_readiness_wait_times_out(
    sample_wav, local_mode, _isolate_side_effects, monkeypatch
):
    """AC 13: a genuinely stuck local load must surface a clear error
    identifying the model as not ready, instead of process_audio hanging (or
    silently falling through into an equally-unbounded transcribe() call)."""
    from app.stt.local import LocalSTTProvider

    def _stuck_get_model(self):
        time.sleep(0.3)

    monkeypatch.setattr(LocalSTTProvider, "_get_model", _stuck_get_model)
    monkeypatch.setattr("app.stt.local_setup._READY_TIMEOUT", 0.05)

    def _boom(*args, **kwargs):
        raise AssertionError("transcribe() must not be reached when the barrier times out")

    monkeypatch.setattr(LocalSTTProvider, "transcribe", _boom)

    with pytest.raises(RuntimeError, match="not become ready"):
        await process_audio(sample_wav, style="normal")


@pytest.mark.asyncio
async def test_process_audio_proceeds_to_transcribe_when_barrier_returns_not_ready_without_timeout(
    sample_wav, local_mode, _isolate_side_effects, monkeypatch
):
    """Orchestrator correctness requirement: a non-timeout `False` from
    await_local_ready() (e.g. ensure_local_ready()'s own early-return guards
    racing in) must NOT block the request -- transcribe()'s own lazy
    _get_model() fallback still applies, exactly as it did before this
    barrier existed. A request must never be worse off than before."""
    from app.stt.base import TranscriptionResult

    _, save_mock = _isolate_side_effects

    async def _fake_await_local_ready(stt_settings, timeout=None):
        return False

    monkeypatch.setattr(
        "app.stt.local_setup.await_local_ready", _fake_await_local_ready
    )

    from app.stt.local import LocalSTTProvider

    monkeypatch.setattr(
        LocalSTTProvider,
        "transcribe",
        AsyncMock(return_value=TranscriptionResult(text="lazy-loaded anyway", tokens_used=None)),
    )

    result = await process_audio(sample_wav, style="normal")

    assert result.text == "lazy-loaded anyway"
    assert save_mock.call_count == 1




@pytest.mark.asyncio
async def test_pipeline_auto_language_substitutes_detected_language(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-21: language="auto" + provider reports "uk" -> saved as "uk"."""
    _, save_mock = _isolate_side_effects
    stt = MagicMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text="привіт", tokens_used=None, detected_language="uk")
    )
    stt.model_name = "mock/provider"
    stt.is_local = False

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        await process_audio(sample_wav, language="auto", style="normal")

    assert save_mock.call_args.kwargs["language"] == "uk"


@pytest.mark.asyncio
async def test_pipeline_explicit_language_never_overridden_by_detection(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-21: language="uk" + provider reports "en" -> saved as "uk" -- an
    explicit user choice is never overridden by a provider's guess."""
    _, save_mock = _isolate_side_effects
    stt = MagicMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text="hello", tokens_used=None, detected_language="en")
    )
    stt.model_name = "mock/provider"
    stt.is_local = False

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        await process_audio(sample_wav, language="uk", style="normal")

    assert save_mock.call_args.kwargs["language"] == "uk"


@pytest.mark.asyncio
async def test_pipeline_auto_language_falls_back_to_auto_sentinel_when_provider_reports_nothing(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-22: language="auto" + provider reports nothing -> saved language
    stays the literal "auto" sentinel -- current behaviour, no regression."""
    _, save_mock = _isolate_side_effects
    stt = _make_stt_mock("ok")

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        await process_audio(sample_wav, language="auto", style="normal")

    assert save_mock.call_args.kwargs["language"] == "auto"




def _silence_analysis(is_silent: bool):
    from app.audio.analysis import SilenceAnalysis

    return SilenceAnalysis(
        peak_dbfs=-60.0 if is_silent else -12.0,
        speech_frame_count=0 if is_silent else 30,
        total_frame_count=33,
        is_silent=is_silent,
    )


def _vad_analysis(is_silent: bool):
    from app.audio.vad import VadAnalysis

    return VadAnalysis(
        speech_frame_count=0 if is_silent else 40,
        total_frame_count=62,
        max_probability=0.12 if is_silent else 0.93,
        is_silent=is_silent,
    )


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_silent_verdict_is_discarded(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-8(a): the loud-non-speech case (clicks, hum, noise) -- the primary
    hole spec 033 closes. A loudness gate cannot see this input at all; since
    spec 034 / ADR 020 the energy guard is not even asked, because the VAD
    verdict decides on its own.

    AC-9's full side-effect guarantee is asserted here, background tasks
    included: today no embedding task can be scheduled because `save_entry`
    is what schedules them, but a refactor that moved scheduling earlier
    would otherwise regress silently on the layer that actually discards."""
    copy_mock, save_mock = _isolate_side_effects
    stt = _make_stt_mock("Дякую за перегляд")

    bt = BackgroundTasks()
    with (
        patch.object(bt, "add_task") as add_task_mock,
        patch("app.pipeline.service.analyze_vad", return_value=_vad_analysis(True)),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(
            sample_wav, language="uk", style="normal", background_tasks=bt
        )

    assert result.discarded_reason == "silence"
    assert result.text == ""
    assert stt.transcribe.await_count == 0
    copy_mock.assert_not_called()
    save_mock.assert_not_called()
    add_task_mock.assert_not_called()


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_speech_verdict_skips_energy_pass_entirely(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """Spec 034 AC-4, the successor to spec 033's AC-8(b).

    This test used to assert an INFO log ("energy guard false positive
    averted by VAD"). Spec 034 / ADR 020 removed that log together with the
    energy pass that fed it: the energy guard is now invoked ONLY when the
    VAD abstains, so on a VAD-speech verdict there is no energy verdict to
    be wrong and nothing to avert. The log's premise stopped existing.

    What survives is the stronger property. The verdict half is unchanged
    (quiet speech the energy guard would have eaten is transcribed), and it
    is now pinned end-to-end against the REAL detectors and the real -20dB
    window by test_vad.py::test_averted_energy_false_positive_is_not_
    discarded_end_to_end, which spec 034 left untouched. On top of that we
    pin the new invariant: energy is not merely overridden, it is never
    called."""
    stt = _make_stt_mock("тихе мовлення")

    with (
        patch("app.pipeline.service.analyze_silence") as energy_mock,
        patch("app.pipeline.service.analyze_vad", return_value=_vad_analysis(False)),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "тихе мовлення"
    assert stt.transcribe.await_count == 1
    assert energy_mock.call_count == 0, (
        "ADR 020 VIOLATED — the energy guard ran despite a VAD verdict being "
        "available. Its verdict would be overridden anyway, and computing it "
        "costs a full-file decode (1707ms on a 6.4min upload)."
    )


@pytest.mark.no_vad_stub
@pytest.mark.parametrize("vad_is_silent", [True, False])
@pytest.mark.asyncio
async def test_energy_pass_is_skipped_whenever_the_vad_has_a_verdict(
    sample_wav, cloud_mode, _isolate_side_effects, vad_is_silent
):
    """AC-1: the lazy-fallback ordering, pinned in BOTH verdict directions.

    A VAD verdict — silent or speech — decides on its own, so the energy
    decode must not be paid at all. Parametrized because a naive
    implementation could easily skip energy on one branch and still run it
    on the other."""
    stt = _make_stt_mock("привіт")

    with (
        patch("app.pipeline.service.analyze_silence") as energy_mock,
        patch("app.pipeline.service.analyze_vad", return_value=_vad_analysis(vad_is_silent)),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert energy_mock.call_count == 0
    assert result.discarded_reason == ("silence" if vad_is_silent else None)


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_energy_pass_runs_exactly_once_when_the_vad_abstains(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-1, the other half: abstention is what makes the fallback fire.

    The no-DLL path (every non-Windows platform, every un-fetched checkout)
    depends on this — there the energy guard is the sole authority, not a
    vestige."""
    stt = _make_stt_mock("привіт")

    with (
        patch(
            "app.pipeline.service.analyze_silence", return_value=_silence_analysis(False)
        ) as energy_mock,
        patch("app.pipeline.service.analyze_vad", return_value=None),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert energy_mock.call_count == 1
    assert result.text == "привіт"


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_absent_falls_back_to_energy_verdict_bit_identically(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-8(c): when the VAD abstains (no binary -- every non-Windows
    platform and every un-fetched checkout), the energy verdict decides
    exactly as shipped in spec 029."""
    stt = _make_stt_mock("Дякую за перегляд")

    with (
        patch("app.pipeline.service.analyze_silence", return_value=_silence_analysis(True)),
        patch("app.pipeline.service.analyze_vad", return_value=None),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        discarded = await process_audio(sample_wav, language="uk", style="normal")

    assert discarded.discarded_reason == "silence"
    assert stt.transcribe.await_count == 0

    stt2 = _make_stt_mock("привіт")
    with (
        patch("app.pipeline.service.analyze_silence", return_value=_silence_analysis(False)),
        patch("app.pipeline.service.analyze_vad", return_value=None),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt2, None)),
    ):
        kept = await process_audio(sample_wav, language="uk", style="normal")

    assert kept.discarded_reason is None
    assert kept.text == "привіт"


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_speech_verdict_transcribes_normally(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-8(d): the overwhelmingly common path -- nothing is discarded.

    Named "both layers pass" until spec 034; since ADR 020 only one layer
    runs here -- the VAD's speech verdict is enough on its own and the energy
    guard is never invoked."""
    stt = _make_stt_mock("привіт світ")

    with (
        patch("app.pipeline.service.analyze_vad", return_value=_vad_analysis(False)),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "привіт світ"
    assert stt.transcribe.await_count == 1


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_discard_logs_deciding_layer_and_measurements(
    sample_wav, cloud_mode, _isolate_side_effects, caplog
):
    """AC-9: the deciding layer must be diagnosable from the WARNING log
    alone -- that is the entire reason all three layers can share
    discarded_reason="silence" without a frontend change."""
    import logging

    stt = _make_stt_mock("Дякую за перегляд")

    with (
        patch("app.pipeline.service.analyze_vad", return_value=_vad_analysis(True)),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
        caplog.at_level(logging.WARNING, logger="app.pipeline.service"),
    ):
        await process_audio(sample_wav, language="uk", style="normal")

    full_log = "\n".join(r.getMessage() for r in caplog.records)
    assert "layer=vad" in full_log
    assert "speech_hops=" in full_log
    assert "max_prob=" in full_log


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_energy_discard_names_its_layer_in_the_log(
    sample_wav, cloud_mode, _isolate_side_effects, caplog
):
    """AC-9: the fallback path is equally diagnosable."""
    import logging

    stt = _make_stt_mock("Дякую за перегляд")

    with (
        patch("app.pipeline.service.analyze_silence", return_value=_silence_analysis(True)),
        patch("app.pipeline.service.analyze_vad", return_value=None),
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
        caplog.at_level(logging.WARNING, logger="app.pipeline.service"),
    ):
        await process_audio(sample_wav, language="uk", style="normal")

    full_log = "\n".join(r.getMessage() for r in caplog.records)
    assert "layer=energy" in full_log


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_disabled_never_calls_analyze_vad(
    sample_wav, cloud_mode, _isolate_side_effects, monkeypatch
):
    """AC-7: silence_vad_enabled=False is a real kill switch -- a field user
    hit by a VAD false positive drops back to shipped-029 behaviour with one
    env var and no rebuild, so the VAD must not even be invoked."""
    stt = _make_stt_mock("привіт")
    monkeypatch.setattr(settings.audio, "silence_vad_enabled", False)

    with (
        patch("app.pipeline.service.analyze_silence", return_value=_silence_analysis(False)),
        patch("app.pipeline.service.analyze_vad") as vad_mock,
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert vad_mock.call_count == 0
    assert result.text == "привіт"


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_disabled_still_discards_on_the_energy_verdict(
    sample_wav, cloud_mode, _isolate_side_effects, monkeypatch
):
    """Spec 034 review YELLOW-1: the kill switch must leave a WORKING gate.

    `test_vad_disabled_never_calls_analyze_vad` above pins only that the VAD
    is not invoked, on a NON-silent energy verdict — so it cannot fail if the
    energy pass stops running too. Spec 034 is the change that made that
    invocation conditional, and this is the branch it made conditional:
    `silence_vad_enabled=False` is exactly what a field user flips when the
    VAD misbehaves, and it must fall back to shipped-029 behaviour, not to no
    gate at all. Narrowing service.py's fallback condition to
    `if vad is None and settings.audio.silence_vad_enabled:` fails here."""
    copy_mock, save_mock = _isolate_side_effects
    stt = _make_stt_mock("Дякую за перегляд")
    monkeypatch.setattr(settings.audio, "silence_vad_enabled", False)

    with (
        patch(
            "app.pipeline.service.analyze_silence", return_value=_silence_analysis(True)
        ) as energy_mock,
        patch("app.pipeline.service.analyze_vad") as vad_mock,
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert vad_mock.call_count == 0
    assert energy_mock.call_count == 1, (
        "the kill switch disabled the VAD *and* the energy guard — with "
        "silence_vad_enabled=False there is no silence gate left at all"
    )
    assert result.discarded_reason == "silence"
    assert result.text == ""
    assert stt.transcribe.await_count == 0
    copy_mock.assert_not_called()
    save_mock.assert_not_called()


@pytest.mark.no_vad_stub
@pytest.mark.asyncio
async def test_vad_runs_off_the_event_loop(sample_wav, cloud_mode, _isolate_side_effects):
    """The VAD is synchronous CPU/IO work on EVERY call, so like
    analyze_silence it must go through asyncio.to_thread -- a slow VAD must
    not stall the event loop (the same YELLOW finding spec 029 fixed)."""
    ticks = {"n": 0}

    def _slow_vad(*args, **kwargs):
        time.sleep(0.25)
        return _vad_analysis(False)

    async def _ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.02)

    stt = _make_stt_mock("привіт")
    task = asyncio.create_task(_ticker())
    try:
        with (
            patch("app.pipeline.service.analyze_vad", _slow_vad),
            patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
        ):
            await process_audio(sample_wav, language="uk", style="normal")
    finally:
        task.cancel()

    assert ticks["n"] > 1, "event loop was blocked during analyze_vad()"




def _stt_mock_with_no_speech(text: str, no_speech_prob):
    stt = MagicMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(text=text, no_speech_prob=no_speech_prob)
    )
    stt.model_name = "mock/provider"
    stt.is_local = False
    return stt


@pytest.mark.asyncio
async def test_high_no_speech_prob_discards_after_model(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-19: the confidently-decoded hallucination -- the exact class
    Whisper's own `no_speech_prob > 0.6 AND avg_logprob < -1.0` never
    suppresses, because the AND fails on a confident decode."""
    copy_mock, save_mock = _isolate_side_effects
    stt = _stt_mock_with_no_speech("Дякую за перегляд!", 0.7)

    bt = BackgroundTasks()
    with (
        patch.object(bt, "add_task") as add_task_mock,
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
    ):
        result = await process_audio(
            sample_wav, language="uk", style="normal", background_tasks=bt
        )

    assert result.discarded_reason == "silence"
    assert result.text == ""
    assert result.copied_to_clipboard is False
    assert result.model_name == "mock/provider"
    copy_mock.assert_not_called()
    save_mock.assert_not_called()
    add_task_mock.assert_not_called()


@pytest.mark.asyncio
async def test_no_speech_prob_exactly_at_threshold_is_kept(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-19: the comparison is a strict `>`, so the boundary value keeps
    the transcription. Fail open on ties."""
    stt = _stt_mock_with_no_speech("реальні слова", 0.6)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "реальні слова"


@pytest.mark.asyncio
async def test_no_speech_prob_none_is_kept(sample_wav, cloud_mode, _isolate_side_effects):
    """AC-19: providers with no signal on the path taken (Gemini always;
    Groq/whisper.cpp on the explicit-language path) must never be discarded
    by a layer that has nothing to say about them."""
    stt = _stt_mock_with_no_speech("реальні слова", None)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "реальні слова"


@pytest.mark.asyncio
async def test_mixed_segments_min_keeps_transcription(
    sample_wav, cloud_mode, _isolate_side_effects
):
    """AC-19: min-across-segments means one confident-speech segment keeps
    the WHOLE result. This is what makes layer 3 maximally conservative on
    real speech by construction -- a hallucinated tail after real words never
    costs the user the real words."""
    stt = _stt_mock_with_no_speech("справжні слова плюс галюцинація", 0.1)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "справжні слова плюс галюцинація"


@pytest.mark.asyncio
async def test_no_speech_discard_logs_layer_and_never_the_text(
    sample_wav, cloud_mode, _isolate_side_effects, caplog
):
    """AC-19: the WARNING names the deciding layer, the threshold and the
    measured value -- but only the LENGTH of the discarded text, never the
    hallucinated text itself."""
    import logging

    secret = "Дякую за перегляд, підпишіться на канал"
    stt = _stt_mock_with_no_speech(secret, 0.95)

    with (
        patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)),
        caplog.at_level(logging.WARNING, logger="app.pipeline.service"),
    ):
        await process_audio(sample_wav, language="uk", style="normal")

    full_log = "\n".join(r.getMessage() for r in caplog.records)
    assert "layer=provider-metadata" in full_log
    assert "0.95" in full_log
    assert secret not in full_log


@pytest.mark.asyncio
async def test_no_speech_threshold_is_settings_driven(
    sample_wav, cloud_mode, _isolate_side_effects, monkeypatch
):
    """AC-19: no_speech_prob_threshold is a real STTSettings field -- raising
    it keeps a result the default would have discarded."""
    stt = _stt_mock_with_no_speech("гранична впевненість", 0.7)
    monkeypatch.setattr(settings.stt, "no_speech_prob_threshold", 0.9)

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)):
        result = await process_audio(sample_wav, language="uk", style="normal")

    assert result.discarded_reason is None
    assert result.text == "гранична впевненість"
