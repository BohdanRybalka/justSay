"""Regression tests for Spec 028 Item 1 -- app-data path isolation.

See docs/adr/014-lazy-app-data-path-resolution.md. Proves the pre-fix bug
(``user_settings.py`` freezing ``SETTINGS_DIR``/``SETTINGS_PATH`` at import
time, defeating every downstream test-isolation fixture) cannot recur,
exercises the conftest.py guard fixture's own detection logic directly, and
(AC 5b) statically verifies the app-data consumer inventory is exhaustive.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import user_settings
from app.main import app
from app.transcripts import history
from tests.conftest import _cleanup_data_dir, _paths_under_real_root, _snapshot_real_roots


def test_testclient_lifespan_keeps_history_under_tmp_path(tmp_path):
    """AC 4 -- the regression test.

    Against pre-fix code, `user_settings.SETTINGS_PATH` is frozen at import
    time to the real ``~/.justsay-dev/settings.json``. Entering
    `TestClient(app)` runs the real `lifespan()`, which reads
    `get_user_settings().output_dir` (defaulting to the frozen constant) and
    calls `history.bootstrap(...)`, re-pointing `history._output_dir` at the
    real directory regardless of this test's own `tmp_path` isolation.

    Revert only the lazy-path change in `user_settings.py` (restore the
    frozen `SETTINGS_DIR`/`SETTINGS_PATH` constants) to see this assertion
    fail -- that is the proof this AC asks for.
    """
    with TestClient(app):
        assert history.history_path().is_relative_to(tmp_path)


def test_testclient_lifespan_keeps_settings_path_under_tmp_path(tmp_path):
    """Companion check: the settings file itself (not just history.db) must
    also resolve under the test's tmp_path once lifespan() has run."""
    with TestClient(app):
        assert user_settings._settings_path().is_relative_to(tmp_path)




def test_guard_helper_reports_no_offenders_when_isolated(tmp_path):
    """Baseline: under normal, correctly-isolated test state (which every
    test in this suite is in, thanks to conftest.py's autouse fixtures), the
    guard helper finds nothing pointing at the real root."""
    real_root = tmp_path / "definitely-not-the-real-root"
    assert _paths_under_real_root([real_root]) == []


def test_guard_helper_detects_history_pointing_at_real_root(tmp_path, monkeypatch):
    """Prove the detection logic actually flags a leak, not just that it
    always happens to pass. Simulates the pre-fix bug directly: repoint
    `history._output_dir` at a stand-in "real root" (a plain tmp_path
    subdirectory, NOT the actual developer root the outer conftest.py guard
    checks against, so this cannot trip that guard's own post-test check)
    and confirm the helper reports it. `monkeypatch.setattr` restores
    `history._output_dir` automatically once this test ends."""
    fake_real_root = tmp_path / "fake-real-root"
    fake_real_root.mkdir()

    monkeypatch.setattr(history, "_output_dir", fake_real_root)
    offenders = _paths_under_real_root([fake_real_root])
    labels = {label for label, _path in offenders}
    assert "history.history_path()" in labels


def test_guard_helper_detects_settings_path_pointing_at_real_root(tmp_path, monkeypatch):
    """Same proof as above, for `user_settings._settings_path()`."""
    fake_real_root = tmp_path / "fake-real-root-2"
    fake_real_root.mkdir()

    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(fake_real_root))
    offenders = _paths_under_real_root([fake_real_root])
    labels = {label for label, _path in offenders}
    assert "user_settings._settings_path()" in labels


def test_guard_helper_detects_audio_temp_dir_pointing_at_real_root(tmp_path, monkeypatch):
    """RED 1's exact bug: `settings.audio.temp_dir` is a third,
    frozen-at-import consumer AC 5's original guard never watched. Prove the
    widened guard now catches it."""
    from app.core.config import settings as runtime_settings

    fake_real_root = tmp_path / "fake-real-root-3"
    fake_real_root.mkdir()

    monkeypatch.setattr(runtime_settings.audio, "temp_dir", fake_real_root / "tmp")
    offenders = _paths_under_real_root([fake_real_root])
    labels = {label for label, _path in offenders}
    assert "settings.audio.temp_dir" in labels


def test_guard_helper_detects_log_file_path_pointing_at_real_root(tmp_path, monkeypatch):
    from app.core import logging_config

    fake_real_root = tmp_path / "fake-real-root-4"
    fake_real_root.mkdir()

    monkeypatch.setattr(logging_config, "_log_file", fake_real_root / "logs" / "backend.log")
    offenders = _paths_under_real_root([fake_real_root])
    labels = {label for label, _path in offenders}
    assert "logging_config.log_file_path()" in labels


def test_guard_helper_detects_crash_guard_path_pointing_at_real_root(tmp_path, monkeypatch):
    fake_real_root = tmp_path / "fake-real-root-5"
    fake_real_root.mkdir()

    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(fake_real_root))
    offenders = _paths_under_real_root([fake_real_root])
    labels = {label for label, _path in offenders}
    assert "local_setup._crash_guard_path()" in labels


def test_guard_helper_checks_both_real_roots_prod_and_dev(tmp_path, monkeypatch):
    """AC 5a: the guard must check a path against EITHER real root (prod or
    dev), not just one -- pass two distinct roots and confirm an offender
    under the second one alone is still caught."""
    prod_root = tmp_path / "not-prod"
    dev_root = tmp_path / "not-dev"
    prod_root.mkdir()
    dev_root.mkdir()

    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(dev_root))
    offenders = _paths_under_real_root([prod_root, dev_root])
    labels = {label for label, _path in offenders}
    assert "user_settings._settings_path()" in labels




def test_snapshot_detects_new_file_in_a_subdirectory_of_a_real_root(tmp_path):
    """The exact gap review pass 2 found. `_snapshot_real_roots` must walk
    recursively -- a file appearing only in `after`, nested under a
    subdirectory of the root, must be visible to the diff."""
    fake_real_root = tmp_path / "fake-real-root-6"
    subdir = fake_real_root / "tmp"
    subdir.mkdir(parents=True)

    before = _snapshot_real_roots([fake_real_root])

    leaked_path = subdir / "leaked.wav"
    leaked_path.write_bytes(b"leaked audio bytes")

    after = _snapshot_real_roots([fake_real_root])

    assert leaked_path not in before
    assert leaked_path in after
    assert after != before


def test_snapshot_survives_a_file_vanishing_between_listing_and_stat(tmp_path, monkeypatch):
    """A transient file disappearing between directory listing and stat()
    (e.g. a pipeline temp file cleaned up mid-walk, or a concurrent sibling
    agent writing/removing its own files under the same real root) must not
    crash the snapshot -- it is skipped, not treated as a leak signal."""
    fake_real_root = tmp_path / "fake-real-root-7"
    subdir = fake_real_root / "tmp"
    subdir.mkdir(parents=True)
    ghost = subdir / "ghost.wav"
    ghost.write_bytes(b"here during listing, gone by stat time")

    real_stat = Path.stat

    def _flaky_stat(self, *a, **kw):
        if self == ghost:
            raise FileNotFoundError("vanished between listing and stat")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", _flaky_stat)

    snapshot = _snapshot_real_roots([fake_real_root])

    assert ghost not in snapshot



_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_EXPECTED_APP_DATA_CONSUMERS: dict[str, str] = {
    "audio/config.py": "frozen-at-import",
    "core/logging_config.py": "lazy",
    "core/user_settings.py": "lazy",
    "stt/local_setup.py": "lazy",
    "stt/local_whisper_cpp_cmd.py": "exception",
    "transcripts/history.py": "lazy",
}

_RESOLVE_CALL_RE = re.compile(r"\bresolve_app_data_root\s*\(|\bresolve_temp_dir\b")
_PATH_HOME_RE = re.compile(r"\bPath\.home\s*\(")


def _scan_app_data_consumers() -> set[str]:
    """Every backend/app/**/*.py file (excluding app_paths.py, the definer)
    that references `resolve_app_data_root()`, `resolve_temp_dir` or a literal
    `Path.home()` -- the three ways a module can resolve an app-data path.

    `resolve_temp_dir` matches without a call-parenthesis because
    `audio/config.py` passes it as a `default_factory` rather than calling it
    (ADR 033); requiring `(` there would silently drop a real consumer."""
    found: set[str] = set()
    for path in _APP_DIR.rglob("*.py"):
        if path.name == "app_paths.py":
            continue
        text = path.read_text(encoding="utf-8")
        if _RESOLVE_CALL_RE.search(text) or _PATH_HOME_RE.search(text):
            found.add(str(path.relative_to(_APP_DIR)).replace("\\", "/"))
    return found


def test_app_data_consumer_inventory_is_exhaustive():
    """AC 5b: a future consumer that resolves an app-data path must be
    registered here (and in docs/adr/014's Consumer inventory table) or this
    test fails -- turning "we forgot to audit" into a red test instead of a
    silent gap. This is a grep-shaped test (it will not see a consumer that
    reaches the root via an alias or an indirection) -- AC 5c's session
    snapshot is the backstop that does not depend on reading source at all.
    """
    discovered = _scan_app_data_consumers()
    expected = set(_EXPECTED_APP_DATA_CONSUMERS)

    missing_from_inventory = discovered - expected
    stale_in_inventory = expected - discovered

    assert not missing_from_inventory, (
        f"Found app-data consumer(s) not registered in the inventory: "
        f"{sorted(missing_from_inventory)}. Add each to "
        f"_EXPECTED_APP_DATA_CONSUMERS (tagged lazy/frozen-at-import/exception) "
        f"and to docs/adr/014-lazy-app-data-path-resolution.md's Consumer "
        f"inventory table."
    )
    assert not stale_in_inventory, (
        f"Inventory lists consumer(s) no longer found by the scan: "
        f"{sorted(stale_in_inventory)}. Remove the stale entry from both "
        f"_EXPECTED_APP_DATA_CONSUMERS and the ADR table."
    )




def test_cleanup_data_dir_releases_the_log_handle_and_removes_the_directory(tmp_path):
    """Reproduces the exact lock: installs a real `RotatingFileHandler`
    tagged `_justsay_file` (matching `logging_config.setup_logging()`'s own
    marker) writing into a throwaway directory -- on Windows, simply
    *opening* that file is enough to lock it, no write required -- then
    calls the actual cleanup helper without pre-closing it, and asserts
    both the handler is detached from the root logger AND the directory is
    actually gone afterwards."""
    import logging.handlers

    session_dir = tmp_path / "session"
    log_file = session_dir / "logs" / "backend.log"
    log_file.parent.mkdir(parents=True)

    handler = logging.handlers.RotatingFileHandler(log_file, encoding="utf-8")
    handler._justsay_file = True
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    handler.emit(
        logging.LogRecord("test", logging.INFO, __file__, 0, "hold the lock", None, None)
    )

    try:
        assert session_dir.is_dir()

        _cleanup_data_dir(session_dir)

        assert handler not in root_logger.handlers, (
            "the _justsay_file handler must be detached before rmtree is attempted"
        )
        assert not session_dir.exists(), (
            "the session directory must actually be removed once the log "
            "handle is released -- this is the Stage 6 regression: "
            "ignore_errors=True let this assertion fail silently forever"
        )
    finally:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


def test_cleanup_data_dir_warns_instead_of_silently_swallowing_a_removal_failure(
    tmp_path, monkeypatch
):
    """If the directory still can't be removed even after the log handle is
    released (some other process has a file open, permissions, ...), that
    must be surfaced -- not swallowed the way `ignore_errors=True` did."""

    def _boom(path, *a, **kw):
        raise OSError("simulated: still locked by something else")

    monkeypatch.setattr("tests.conftest.shutil.rmtree", _boom)

    with pytest.warns(UserWarning, match="Failed to remove"):
        _cleanup_data_dir(tmp_path / "does-not-matter")
