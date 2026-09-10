import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import pytest

from app.audio.dependencies import get_recorder
from app.main import app
from app.pipeline.service import ProcessingResult


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"]
    assert data["stt_mode"] in ("cloud", "local")
    assert data["llm_mode"] in ("cloud", "local")


@pytest.mark.asyncio
async def test_set_stt_mode_accepts_json_object(client):
    """Wire format ``{"mode": "..."}`` must keep working after ProviderModeUpdate removal."""
    resp = await client.put("/stt/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "local"

    resp = await client.put("/stt/mode", json={"mode": "cloud"})
    assert resp.status_code == 200
    assert resp.json()["stt_mode"] == "cloud"


@pytest.mark.asyncio
async def test_switch_stt_mode_invalid(client):
    resp = await client.put("/stt/mode", json={"mode": "quantum"})
    assert resp.status_code == 422




@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_set_stt_mode_triggers_prewarm_without_awaiting_it(client, monkeypatch):
    """`PUT /stt/mode` must call `maybe_prewarm_local` exactly once when
    switching to "local", and the response must return before a
    deliberately slow monkeypatched `maybe_prewarm_local` completes —
    confirming the trigger is genuinely fire-and-forget, not awaited inline.

    Marked `@pytest.mark.prewarm` so backend/tests/conftest.py's autouse
    no-op fixture doesn't mask this call — this test patches
    `maybe_prewarm_local` itself before any request is made, so the real
    `ensure_local_ready` (which could attempt a real pip install) never runs.
    """
    import app.stt.local_setup as local_setup_module

    call_count = {"n": 0}
    never_set = asyncio.Event()
    background: dict = {}

    def _slow_spy(stt_settings):
        call_count["n"] += 1

        async def _block_forever():
            await never_set.wait()

        background["task"] = asyncio.create_task(_block_forever())

    monkeypatch.setattr(local_setup_module, "maybe_prewarm_local", _slow_spy)

    resp = await client.put("/stt/mode", json={"mode": "local"})

    assert resp.status_code == 200
    assert call_count["n"] == 1
    assert not background["task"].done()

    never_set.set()
    await background["task"]


@pytest.mark.asyncio
async def test_stt_local_prewarm_rejects_when_not_local_mode(client):
    """Default STT mode is Cloud (reset by conftest's `_reset_settings`
    fixture) — the endpoint must 400 without ever touching `maybe_prewarm_local`."""
    resp = await client.post("/stt/local/prewarm")
    assert resp.status_code == 400
    assert "not local" in resp.json()["detail"].lower()


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_stt_local_prewarm_dispatches_and_returns_started(client, monkeypatch):
    """`POST /stt/local/prewarm` dispatches `maybe_prewarm_local` and returns
    `{"started": true}` without awaiting completion.

    Marked `@pytest.mark.prewarm`; `maybe_prewarm_local` is patched to a
    counting spy *before* the mode switch, so the real function (and any
    real pip install it could trigger) never runs at any point in this test.
    """
    import app.stt.local_setup as local_setup_module

    call_count = {"n": 0}

    def _spy(stt_settings):
        call_count["n"] += 1

    monkeypatch.setattr(local_setup_module, "maybe_prewarm_local", _spy)

    resp = await client.put("/stt/mode", json={"mode": "local"})
    assert resp.status_code == 200
    assert call_count["n"] == 1

    resp = await client.post("/stt/local/prewarm")
    assert resp.status_code == 200
    assert resp.json() == {"started": True}
    assert call_count["n"] == 2


def _make_recorder_mock(duration: float, audio_path: Path) -> MagicMock:
    recorder = MagicMock()
    recorder.is_recording = True
    recorder.stop = AsyncMock(return_value=audio_path)
    recorder.last_duration_seconds = duration
    return recorder


def _make_pipeline_result() -> ProcessingResult:
    return ProcessingResult(
        text="ok", duration_ms=100, copied_to_clipboard=True
    )


@pytest.mark.asyncio
async def test_dictate_zero_duration_passes_none_to_pipeline(client, tmp_path):
    """0.0s captured_duration must not be forwarded — pipeline should re-detect."""
    audio_file = tmp_path / "rec.wav"
    audio_file.write_bytes(b"")

    recorder = _make_recorder_mock(duration=0.0, audio_path=audio_file)

    app.dependency_overrides[get_recorder] = lambda: recorder
    with patch(
        "app.pipeline.router.process_audio", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = _make_pipeline_result()
        await client.post("/pipeline/dictate")

    assert mock_process.call_args.kwargs["audio_duration"] is None


@pytest.mark.asyncio
async def test_dictate_positive_duration_forwarded_to_pipeline(client, tmp_path):
    """Positive captured_duration must be forwarded as-is to avoid re-detection."""
    audio_file = tmp_path / "rec.wav"
    audio_file.write_bytes(b"")

    recorder = _make_recorder_mock(duration=7.5, audio_path=audio_file)

    app.dependency_overrides[get_recorder] = lambda: recorder
    with patch(
        "app.pipeline.router.process_audio", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = _make_pipeline_result()
        await client.post("/pipeline/dictate")

    assert mock_process.call_args.kwargs["audio_duration"] == 7.5


@pytest.mark.asyncio
async def test_get_recorder_raises_when_app_state_unset(client):
    """get_recorder()'s RuntimeError guard (Spec 009's core migration
    guarantee: no fallback-constructed recorder, ever) fires when no
    dependency_overrides is set and app.state.recorder was never populated —
    the `client` fixture's ASGITransport does not run the app's lifespan
    context, so this is the default state of every test in this suite that
    doesn't explicitly override get_recorder."""
    with pytest.raises(RuntimeError, match="app.state.recorder is not set"):
        await client.get("/audio/status")



class _FakeStreamingRecorder:
    """Fake recorder whose is_recording flips False after N reads — the
    stream's termination is deterministic on the fake's own read-count
    exhaustion, not on request.is_disconnected()."""

    def __init__(self, recording_reads: int, level_db: float = -12.3):
        self._remaining = recording_reads
        self.level_db = level_db

    @property
    def is_recording(self) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False


@pytest.mark.asyncio
async def test_level_stream_emits_level_frames_then_done(client):
    app.dependency_overrides[get_recorder] = lambda: _FakeStreamingRecorder(
        recording_reads=2
    )
    async with client.stream("GET", "/audio/level-stream") as resp:
        assert resp.status_code == 200
        body = b""
        async for chunk in resp.aiter_bytes():
            body += chunk

    assert body.count(b"event: level") == 2
    assert body.count(b"event: done") == 1


@pytest.mark.asyncio
async def test_level_stream_not_recording_emits_only_done(client):
    app.dependency_overrides[get_recorder] = lambda: _FakeStreamingRecorder(
        recording_reads=0
    )
    async with client.stream("GET", "/audio/level-stream") as resp:
        assert resp.status_code == 200
        body = b""
        async for chunk in resp.aiter_bytes():
            body += chunk

    assert b"event: level" not in body
    assert body.count(b"event: done") == 1


@pytest.mark.asyncio
async def test_history_limit_is_bounded(client):
    """`GET /history` returns full rows and had no bound at all, while both
    siblings in the same package clamp theirs — `/history/search` at
    SEARCH_LIMIT_MAX and `/words/top` at TOP_LIMIT_MAX. The largest real caller
    asks for 50 (`src/settings/tabs/metrics.ts`)."""
    from app.transcripts.history import HISTORY_LIMIT_MAX

    assert (await client.get(f"/history?limit={HISTORY_LIMIT_MAX}")).status_code == 200
    assert (await client.get(f"/history?limit={HISTORY_LIMIT_MAX + 1}")).status_code == 422
    assert (await client.get("/history?limit=0")).status_code == 422



@pytest.mark.asyncio
async def test_history_rejects_half_a_cursor(client):
    """A cursor is a position in ``(ts DESC, id DESC)``, so half of one names
    nothing. FastAPI cannot say "both or neither" in a signature, which is why the
    handler checks it and answers 422 rather than quietly paging from the top.

    An empty ``before_id`` is a whole cursor, not half of one: ``(ts, "")`` sorts
    below every real id at that ``ts``, so it names a well-defined position and
    answers 200, exactly as a cursor naming a row that never existed does."""
    assert (await client.get("/history?limit=30&before_ts=1700000000000")).status_code == 422
    assert (await client.get("/history?limit=30&before_id=abc")).status_code == 422
    assert (await client.get("/history?limit=30&before_ts=abc&before_id=x")).status_code == 422
    assert (
        await client.get("/history?limit=30&before_ts=1700000000000&before_id=nosuchrow")
    ).status_code == 200
    assert (
        await client.get("/history?limit=30&before_ts=1700000000000&before_id=")
    ).status_code == 200


@pytest.mark.asyncio
async def test_history_rejects_a_before_ts_sqlite_cannot_hold(client):
    """``ts`` is a SQLite INTEGER, so a wider value raises ``OverflowError`` out of
    the driver, past ``store_busy_as_503``, and answers 500 with a traceback. The
    bound is in the signature, so it is 422. ``before_id`` carries no such bound and
    needs none -- see ``test_a_cursor_the_server_minted_is_accepted_back_however_long``."""
    from app.transcripts.history import CURSOR_TS_MAX, CURSOR_TS_MIN

    assert (
        await client.get(f"/history?limit=30&before_ts={CURSOR_TS_MAX}&before_id=x")
    ).status_code == 200
    assert (
        await client.get(f"/history?limit=30&before_ts={CURSOR_TS_MIN}&before_id=x")
    ).status_code == 200
    assert (
        await client.get(f"/history?limit=30&before_ts={CURSOR_TS_MAX + 1}&before_id=x")
    ).status_code == 422
    assert (
        await client.get(f"/history?limit=30&before_ts={CURSOR_TS_MIN - 1}&before_id=x")
    ).status_code == 422


@pytest.mark.asyncio
async def test_history_pages_over_http_without_repeating_an_entry(client):
    """The whole contract end to end: two page-30 requests over a 60-entry store
    with a dictation saved between them return 60 distinct ids, and the client
    only ever echoes the cursor it was handed."""
    from app.transcripts import history

    for index in range(60):
        history.save_entry(text=f"entry {index}", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute("BEGIN")
        try:
            for offset, row in enumerate(
                conn.execute("SELECT id FROM entries ORDER BY rowid").fetchall()
            ):
                conn.execute(
                    "UPDATE entries SET ts = ? WHERE id = ?", (1_700_000_000_000 + offset, row[0])
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    first = (await client.get("/history?limit=30")).json()
    assert first["total"] == 60
    assert first["next_cursor"] is not None

    history.save_entry(text="landed between the two pages", duration_ms=1)

    cursor = first["next_cursor"]
    second = (
        await client.get(
            f"/history?limit=30&before_ts={cursor['ts']}&before_id={cursor['id']}"
        )
    ).json()

    ids = [e["id"] for e in first["entries"]] + [e["id"] for e in second["entries"]]
    assert len(ids) == 60
    assert len(set(ids)) == 60
    assert second["next_cursor"] is None
    assert second["total"] == 61


@pytest.mark.asyncio
async def test_a_cursor_the_server_minted_is_accepted_back_however_long(
    client, insert_history_rows
):
    """ADR 053's echo over HTTP, on an id this build could not have minted.

    An adopted history holds ids of any length, so ``next_cursor`` can carry one and
    the client echoes back exactly what it was handed: a length bound on
    ``before_id`` answered the app's own cursor with 422 and ended paging one page
    in. ``before_ts`` keeps its bound, which no minted ``ts`` can fail.

    The id is percent-encoded on the way back because that is what the real client
    does (``src/api.ts`` calls ``encodeURIComponent``); interpolating it raw would
    agree with the app only for ids free of ``&``, ``+`` and ``#``.
    """
    adopted_id = "x" * 100
    insert_history_rows(
        [
            ("newest", 1_700_000_000_002, "row"),
            (adopted_id, 1_700_000_000_001, "row"),
            ("oldest", 1_700_000_000_000, "row"),
        ]
    )

    first = await client.get("/history?limit=2")

    assert first.status_code == 200, first.text
    cursor = first.json()["next_cursor"]
    assert cursor is not None
    assert cursor["id"] == adopted_id

    second = await client.get(
        f"/history?limit=2&before_ts={cursor['ts']}&before_id={quote(cursor['id'], safe='')}"
    )

    assert second.status_code == 200, second.text
    assert [entry["id"] for entry in second.json()["entries"]] == ["oldest"]
    assert second.json()["next_cursor"] is None


@pytest.mark.asyncio
async def test_an_unbounded_before_id_is_data_not_a_control(client, insert_history_rows):
    """Dropping ``before_id``'s length bound removed no protection that was real.

    The bound looked like a guard on untrusted input, so its removal has to be
    pinned rather than asserted: the value is a bound parameter in a comparison,
    never SQL, and its length is not a failure mode. An 8 KB id and one made of SQL
    metacharacters both name a position no row occupies, which is a well-defined
    question with a well-defined answer -- 200 and the rows below it, per ADR 053 --
    and the table is intact afterwards.
    """
    insert_history_rows(
        [("newest", 1_700_000_000_002, "row"), ("oldest", 1_700_000_000_001, "row")]
    )

    for hostile_id in ("x" * 8192, "'; DROP TABLE entries; --", "a&b+c#d"):
        response = await client.get(
            f"/history?limit=30&before_ts=1700000000003&before_id={quote(hostile_id, safe='')}"
        )

        assert response.status_code == 200, (hostile_id[:40], response.text)
        assert [entry["id"] for entry in response.json()["entries"]] == ["newest", "oldest"]

    assert (await client.get("/history?limit=30")).json()["total"] == 2
