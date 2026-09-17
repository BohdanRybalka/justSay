"""The lifespan startup half: which steps may fail without ending the process.

Decided 2026-09-15 as open question E2 and written into docs/style-guide.md
3.3. The shutdown half already had this shape, so these are the startup
mirror of the teardown tests in tests/test_background_tasks.py.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import app.main
from app.main import app as fastapi_app
from app.preferences import user_settings
from app.stt import local_setup
from app.transcripts import history

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


def test_a_failing_history_bootstrap_leaves_the_backend_serving(monkeypatch, caplog):
    """The user-visible half. An unreadable history file used to end the
    process, and the shell then reported only "Backend process exited
    unexpectedly. Check Python dependencies." (src-tauri/src/backend.rs:1136).
    """
    monkeypatch.setattr(history, "bootstrap", _raise)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200

    assert any("opening the history store" in r.getMessage() for r in caplog.records)


def test_history_opens_lazily_after_a_failed_bootstrap(monkeypatch):
    """Skipping bootstrap is a defined path rather than a hole, which is what
    makes this step guardable: `_ensure_conn_locked` opens the connection on
    first use, so the store answers once the app is running.
    """
    monkeypatch.setattr(history, "bootstrap", _raise)

    with TestClient(fastapi_app):
        assert history.get_page(limit=1).entries == []


def test_a_failing_scratch_repair_leaves_the_backend_serving(monkeypatch, caplog):
    """`repair_scratch_output_dir` runs before the store is opened and already
    survives its own failed merge internally; raising outward must not be
    worse than the outcome it handles.
    """
    monkeypatch.setattr(user_settings, "repair_scratch_output_dir", _raise)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(fastapi_app) as client:
            assert client.get("/health").status_code == 200

    assert any("opening the history store" in r.getMessage() for r in caplog.records)


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
