"""The lifespan startup half: which steps may fail without ending the process.

Decided 2026-09-15 as open question E2 and written into docs/style-guide.md
3.3. The shutdown half already had this shape, so these are the startup
mirror of the teardown tests in tests/test_background_tasks.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.audio.meeting_recorder
import app.audio.recorder
import app.embeddings
import app.main
import app.stt
from app.audio.dependencies import get_recorder
from app.core.app_paths import resolve_app_data_root
from app.main import app as fastapi_app
from app.preferences import user_settings
from app.preferences.user_settings import UserSettings
from app.stt import local_setup
from app.transcripts import history, schema

STEP_FAILURE = "startup step blew up"


@pytest.fixture(autouse=True)
def _no_real_gpu_probe(monkeypatch):
    """`lifespan()` is left real for this module, and conftest no-ops the
    prewarm and the background indexer but not `probe_gpu` -- an unpatched
    startup runs a genuine GPU probe in a worker thread on every test here.
    """

    async def _noop() -> None:
        return None

    monkeypatch.setattr(app.main, "_warm_gpu_probe_cache", _noop)


def _raise(*_args, **_kwargs):
    raise RuntimeError(STEP_FAILURE)


def _pin_output_dir(monkeypatch, directory: Path) -> None:
    """Point the stored settings at `directory` through the real
    `get_user_settings()`, which reads this cache when it is warm.

    conftest's autouse `_isolated_app_data` clears the cache before every
    test and monkeypatch restores that `None` afterwards, so nothing here
    outlives the test.
    """
    monkeypatch.setattr(
        user_settings, "_settings", UserSettings(output_dir=str(directory))
    )


class _IdleRecorder:
    """The `get_recorder` override `/audio/start` needs, and nothing else.

    State is per instance, and the override below hands out one instance for
    the whole test -- the shape `get_recorder` has in the real app, where
    `app.state.recorder` outlives every request. A class-attribute stub whose
    `start()` writes instance attributes reads as shared state while being
    per-instance, and a factory minting one per request would lose the
    `is_recording` the previous request set.
    """

    def __init__(self) -> None:
        self.is_recording = False
        self.duration_seconds = 0.0
        self.level_db = -60.0
        self.session_id: str | None = None

    async def start(self, session_id: str | None = None) -> None:
        self.is_recording = True
        self.session_id = session_id


def test_a_failing_history_bootstrap_leaves_the_backend_serving(monkeypatch, caplog):
    """The user-visible half. An unreadable history file used to end the
    process, and the shell then reported only "Backend process exited
    unexpectedly. Check Python dependencies." (src-tauri/src/backend.rs:1136).
    """
    monkeypatch.setattr(history, "bootstrap", _raise)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200

    warnings = [r.getMessage() for r in caplog.records]
    assert any("opening the history store" in message for message in warnings)
    assert not any("repairing the history location" in message for message in warnings)


def test_a_failing_repair_opens_the_store_at_the_configured_directory(
    tmp_path, monkeypatch
):
    """The defect the GitHub review of PR #127 found. Guarding the repair and
    the store open as one step meant a failed repair skipped `bootstrap`,
    left `history._output_dir` unset, and sent every read and write to
    `resolve_app_data_root()` -- an empty History on screen and that
    session's transcripts written where nothing reads them.
    """
    configured = tmp_path / "configured-history"
    history.bootstrap(configured)
    seeded = history.save_entry(text="rows the user can already see", duration_ms=1)
    with history._lock:
        history._close_conn_locked()
        history._output_dir = None

    _pin_output_dir(monkeypatch, configured)
    monkeypatch.setattr(user_settings, "repair_scratch_output_dir", _raise)

    with TestClient(fastapi_app) as client:
        assert history.history_path().parent == configured
        assert history.history_path().parent != resolve_app_data_root()
        response = client.get("/history")
        assert response.status_code == 200
        assert [entry["id"] for entry in response.json()["entries"]] == [seeded.id]


def test_a_moved_history_is_read_at_its_new_home_when_the_setting_cannot_be_stored(
    tmp_path, monkeypatch
):
    """The other half of the same defect, and what the fallback above is for.

    `consolidate_into` merges the rows into the app-data root and renames the
    source `history.db` aside, so the moment it succeeds the stored
    `output_dir` names a directory with no database left in it. Writing the
    new one can still fail -- a read-only settings file, a full disk -- and
    the repair must not read as failed when it does: the fallback would then
    open an empty store in the scratch tree, show an empty History and write
    this session's transcripts where the next Clear Temp Files sweep is
    aimed. The repair runs again on the next launch, where the merge is a
    no-op.
    """
    scratch_history = tmp_path / "tmp" / "history"
    scratch_history.mkdir(parents=True)
    history.bootstrap(scratch_history)
    history.save_entry(text="rows the user can already see", duration_ms=1)
    with history._lock:
        history._close_conn_locked()
        history._output_dir = None

    _pin_output_dir(monkeypatch, scratch_history)
    monkeypatch.setattr(user_settings, "_save", _raise)

    with TestClient(fastapi_app) as client:
        assert history.history_path().parent == resolve_app_data_root()
        assert Path(user_settings.get_user_settings().output_dir) == scratch_history
        assert not (scratch_history / history.HISTORY_FILENAME).exists()
        response = client.get("/history")
        assert response.status_code == 200
        assert [entry["text"] for entry in response.json()["entries"]] == [
            "rows the user can already see"
        ]


def test_a_failed_schema_init_leaves_no_half_open_connection(tmp_path, monkeypatch):
    """`bootstrap` assigns the connection and only then applies the schema,
    while `_ensure_conn_locked` heals a store that has no connection at all.
    A connection left behind by a failed schema step is therefore handed to
    every read and write for the rest of the process -- writes into a
    half-migrated database, where the process used to die instead.
    """
    configured = tmp_path / "half-migrated-history"
    _pin_output_dir(monkeypatch, configured)
    real_init_schema = schema._init_schema
    attempts: list[object] = []

    def _fail_the_first_attempt(conn):
        attempts.append(conn)
        if len(attempts) == 1:
            raise RuntimeError(STEP_FAILURE)
        return real_init_schema(conn)

    monkeypatch.setattr(schema, "_init_schema", _fail_the_first_attempt)

    with TestClient(fastapi_app) as client:
        assert history._conn is None
        assert client.get("/history").status_code == 200

    assert len(attempts) == 2


def test_a_failing_scratch_repair_leaves_the_backend_serving(monkeypatch, caplog):
    """`repair_scratch_output_dir` runs before the store is opened and already
    survives its own failed merge internally; raising outward must not be
    worse than the outcome it handles. It is its own named step, so the
    warning must not blame the store open that did in fact run.
    """
    monkeypatch.setattr(user_settings, "repair_scratch_output_dir", _raise)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200

    warnings = [r.getMessage() for r in caplog.records]
    assert any("repairing the history location" in message for message in warnings)
    assert not any("opening the history store" in message for message in warnings)


def test_an_unreadable_store_answers_the_request_instead_of_ending_the_process(
    tmp_path, monkeypatch
):
    """The honest outcome for a store that genuinely cannot be opened, with
    `bootstrap` left real: the process serves, and the cost lands on the
    History requests themselves rather than on the whole app.

    500 rather than a typed 503 -- `store_busy_as_503` maps only the "locked"
    marker and nothing registers a handler for a bare
    `sqlite3.OperationalError`. Deliberate and cut to a future spec; what is
    pinned here is that the request is answered at all, twice.
    """
    configured = tmp_path / "unreadable-history"
    configured.mkdir()
    (configured / history.HISTORY_FILENAME).mkdir()
    _pin_output_dir(monkeypatch, configured)

    with TestClient(fastapi_app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200
        assert 500 <= client.get("/history").status_code < 600
        assert 500 <= client.get("/history").status_code < 600


def test_a_failing_prewarm_leaves_the_backend_serving(monkeypatch, caplog):
    """`maybe_prewarm_local_at_startup` is best-effort by its own design and
    reads a crash-guard counter file, so an unreadable counter must not cost
    the backend.
    """
    monkeypatch.setattr(local_setup, "maybe_prewarm_local_at_startup", _raise)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200

    assert any("prewarming the local model" in r.getMessage() for r in caplog.records)


def test_a_failing_meeting_recorder_leaves_dictation_working(monkeypatch, caplog):
    """Meeting recording is not a shipped feature and its consumers already
    model its absence, so the app is useful without it: dictation, History
    and Settings all work, and only the meeting endpoints fail.

    Shutdown is pinned in the same test because an unbuilt meeting recorder
    used to break the release block while the step tuple was being *built* --
    outside every guard -- taking the STT and embeddings caches with it, and
    with them `recorder.cleanup()`: the one release step with a real OS
    resource behind it, and so the one asserted here by name.

    Nothing was ever built to release, so the release list leaves the
    meeting recorder out rather than warning about it: one failure logs one
    warning, at the step that actually failed.
    """
    released: list[str] = []
    monkeypatch.setattr(app.audio.meeting_recorder, "MeetingRecorder", _raise)
    monkeypatch.setattr(app.stt, "clear_cache", lambda: released.append("stt"))
    monkeypatch.setattr(
        app.embeddings, "clear_cache", lambda: released.append("embeddings")
    )
    monkeypatch.setattr(
        app.audio.recorder.MicrophoneRecorder,
        "cleanup",
        lambda self: released.append("recorder"),
    )
    idle_recorder = _IdleRecorder()
    fastapi_app.dependency_overrides[get_recorder] = lambda: idle_recorder

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200
            assert not hasattr(fastapi_app.state, "meeting_recorder")
            assert client.post("/audio/start").status_code == 200

    warnings = [r.getMessage() for r in caplog.records]
    assert any("building the meeting recorder" in message for message in warnings)
    assert not any("releasing the meeting recorder" in message for message in warnings)
    assert released == ["stt", "embeddings", "recorder"]


def test_a_failing_sync_to_runtime_still_takes_the_backend_down(monkeypatch):
    """The other half of the rule. Running on .env defaults while the user
    believes their saved settings applied is silently wrong, which is worse
    than not starting.
    """
    monkeypatch.setattr(user_settings, "sync_to_runtime", _raise)

    with pytest.raises(RuntimeError, match=STEP_FAILURE):
        with TestClient(fastapi_app):
            pass


def test_unreadable_user_settings_still_take_the_backend_down(monkeypatch):
    """`get_user_settings` feeds `sync_to_runtime`, so it is fatal for the same
    reason: there is no settings object to apply and no honest default.
    """
    monkeypatch.setattr(user_settings, "get_user_settings", _raise)

    with pytest.raises(RuntimeError, match=STEP_FAILURE):
        with TestClient(fastapi_app):
            pass


def test_an_unwritable_temp_dir_still_takes_the_backend_down(tmp_path, monkeypatch):
    """The scratch directory holds every capture on its way to the STT
    provider, so a backend that cannot create it can transcribe nothing.

    The failure is the real one rather than a patched `mkdir`: the parent of
    the configured scratch directory is an ordinary file, which is what a
    user pointing the app at one looks like from here.
    """
    blocker = tmp_path / "a-file-where-a-directory-should-be"
    blocker.write_text("not a directory")
    monkeypatch.setattr(app.main.settings.audio, "temp_dir", blocker / "tmp")

    with pytest.raises(OSError):
        with TestClient(fastapi_app):
            pass


def test_a_failing_microphone_recorder_still_takes_the_backend_down(monkeypatch):
    """Without it there is no dictation, which is the shipped product -- the
    same test docs/style-guide.md 3.3 applies to every lifespan step.
    """
    monkeypatch.setattr(app.audio.recorder, "MicrophoneRecorder", _raise)

    with pytest.raises(RuntimeError, match=STEP_FAILURE):
        with TestClient(fastapi_app):
            pass


def test_a_guarded_step_does_not_skip_the_steps_after_it(monkeypatch):
    """The startup mirror of
    test_lifespan_release_steps_do_not_skip_each_other_on_failure: a failure in
    the first guarded step must not cost the settings sync or the recorders.
    """
    monkeypatch.setattr(history, "bootstrap", _raise)
    synced: list[str] = []
    real_sync = user_settings.sync_to_runtime

    def _recording_sync(us):
        synced.append("sync")
        return real_sync(us)

    monkeypatch.setattr(user_settings, "sync_to_runtime", _recording_sync)

    with TestClient(fastapi_app):
        assert synced == ["sync"]
        assert hasattr(fastapi_app.state, "recorder")
        assert hasattr(fastapi_app.state, "meeting_recorder")
