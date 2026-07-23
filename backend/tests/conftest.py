import asyncio
import os
import tempfile

# Spec 028 Item 1 (iteration-2 review fix, RED 1). This MUST be the very
# first thing this module does -- before `from app.main import app` and
# before every other `app.*` import, including transitive ones. Laziness
# alone (the iteration-1 fix) is not enough: a value can be resolved lazily
# and still be frozen once, early. `AudioSettings.temp_dir`
# (backend/app/audio/config.py:14) uses a `default_factory`, but the
# `settings = AppSettings()` singleton (backend/app/core/config.py:40)
# invokes it exactly ONCE at import time, during pytest collection, before
# any fixture can exist -- so a per-test `monkeypatch.setenv` fixture can
# never reach it; there is nothing to intercept, only an absence of a call
# during the test. Setting the env var here, before the first `app.*`
# import, makes the real root unreachable for the WHOLE process: every
# import-time resolution -- found or not yet found, audited or not --
# freezes against this throwaway directory instead. See
# docs/adr/014-lazy-app-data-path-resolution.md.
_SESSION_DATA_DIR = tempfile.mkdtemp(prefix="justsay-pytest-")
os.environ["JUSTSAY_DATA_DIR"] = _SESSION_DATA_DIR

# TrustedHostMiddleware (spec 040) rejects any Host outside settings.trusted_hosts
# with 400. The test clients reach the app over base_url "http://test" (Host:
# test) and Starlette's TestClient uses "testserver", so both must be allowed.
# Set here -- before the first `app.*` import below constructs the AppSettings()
# singleton -- because JUSTSAY_TRUSTED_HOSTS is only read at that construction,
# mirroring the JUSTSAY_DATA_DIR seam above. See
# docs/adr/026-loopback-api-request-authentication.md.
os.environ["JUSTSAY_TRUSTED_HOSTS"] = (
    '["127.0.0.1", "localhost", "test", "testserver"]'
)

import logging
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import history, user_settings
from app.core.app_paths import DEV_DIR_NAME, PROD_DIR_NAME
from app.core.config import settings
from app.core.gpu_probe import clear_cache as clear_gpu_probe_cache
from app.main import app
from app.stt import clear_cache as clear_stt_cache


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class _SpawnSpy:
    """Records each call `app.core.tasks.spawn_background_task()` makes while
    patched in: the `name` it was given, and the real `asyncio.Task` object
    it returned. `.tasks` exists specifically so a test can await the exact
    tasks it spawned rather than everything pending on the loop (Spec 032,
    GitHub review finding 1)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.tasks: list[asyncio.Task] = []


@pytest.fixture
def spawn_spy(monkeypatch):
    """Spy on `app.core.tasks.spawn_background_task`, still calling through
    to the real implementation (Spec 032, GitHub review finding 3 -- shared
    fixture instead of a `_spy` closure copy-pasted at each call site).

    `main.py` and `local_setup.py` both do `from app.core import tasks` and
    call `tasks.spawn_background_task(...)` -- there is only one
    `app.core.tasks` module object, so patching its attribute here is visible
    through either module's `tasks` reference.
    """
    from app.core import tasks

    spy = _SpawnSpy()
    real_spawn = tasks.spawn_background_task

    def _spy(coro, *, name):
        task = real_spawn(coro, name=name)
        spy.names.append(name)
        spy.tasks.append(task)
        return task

    monkeypatch.setattr(tasks, "spawn_background_task", _spy)
    return spy


# --- Spec 028 Item 1: app-data path isolation -------------------------------
#
# See docs/adr/014-lazy-app-data-path-resolution.md. `JUSTSAY_DATA_DIR` is the
# one supported test/dev seam for redirecting the data root -- it is checked
# first in `resolve_app_data_root()`, so it wins regardless of import order or
# what has already been imported. Patching `Path.home()` (the mechanism
# several older per-file fixtures used) is NOT supported: it is a no-op
# against any module-level constant that was already resolved before the
# patch was applied -- exactly the Spec 028 Item 1 bug this isolation closes.


def _cleanup_data_dir(path: str | Path) -> None:
    """Release the logging file handle, then remove `path`.

    Stage 6 tester finding: `shutil.rmtree(path, ignore_errors=True)` alone
    silently failed on Windows, every single run -- `setup_logging()`'s
    `RotatingFileHandler` (tagged `_justsay_file`) keeps `backend.log` open
    for the life of the process, which keeps the file (and therefore the
    whole directory) locked. `ignore_errors=True` swallowed the resulting
    `PermissionError` completely -- 44+ leftover `justsay-pytest-*`
    directories accumulated under the OS temp dir before this was caught.
    That silent-failure shape is exactly what this spec exists to
    eliminate, so the fix is releasing the actual lock, not hiding the
    error left over after failing to.

    Detaching and closing the handler (rather than the blunter
    `logging.shutdown()`, which would tear down every registered logger
    process-wide, including ones unrelated to this directory) releases the
    OS file handle. If removal still fails afterwards for some OTHER reason
    (another process has a file open, permissions), that is surfaced via
    `warnings.warn` -- naming the directory and the underlying error --
    rather than swallowed.
    """
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_justsay_file", False):
            root_logger.removeHandler(handler)
            handler.close()

    try:
        shutil.rmtree(path)
    except OSError as e:
        warnings.warn(
            f"Failed to remove the throwaway pytest session data directory "
            f"{str(path)!r} after releasing the logging handle: {e}. It "
            f"will be left behind under the OS temp directory.",
            stacklevel=2,
        )


@pytest.fixture(scope="session", autouse=True)
def _cleanup_session_data_dir():
    """Session finalizer for the throwaway directory `JUSTSAY_DATA_DIR` was
    pinned to above. Without this, a dev machine that runs the suite dozens
    of times a day accumulates one leftover directory (containing a real
    backend.log, and possibly a history.db) per run under the OS temp dir."""
    yield
    _cleanup_data_dir(_SESSION_DATA_DIR)


@pytest.fixture(scope="session", autouse=True)
def _real_app_data_roots() -> tuple[Path, Path]:
    """The two real app-data roots (prod + dev) that must stay unreachable
    for the whole session -- the reference points the guard/snapshot
    fixtures below check every test (and the session) against.

    Computed directly from `Path.home() / PROD_DIR_NAME|DEV_DIR_NAME`, NOT
    via `resolve_app_data_root()`: that function now *always* returns the
    throwaway session directory, since `JUSTSAY_DATA_DIR` was set at this
    module's import time, above -- asking it for "the real root" would just
    hand back the fake one.
    """
    home = Path.home()
    return (home / PROD_DIR_NAME, home / DEV_DIR_NAME)


@pytest.fixture(autouse=True)
def _isolated_app_data(tmp_path, monkeypatch):
    """Redirects every app-data path (settings.json, history.db, the
    frozen-at-import `settings.audio.temp_dir` singleton) under a per-test
    `tmp_path`, and resets every cache that could otherwise survive from a
    previous test into this one.

    Must run before the `client` fixture and before any `TestClient(app)`
    construction -- guaranteed here because it's declared in conftest.py
    (collected before same-scope fixtures declared in test modules) and is
    autouse (autouse fixtures run before explicitly-requested fixtures of the
    same scope).
    """
    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(tmp_path))
    # settings.audio.temp_dir was already frozen at process start (see the
    # module-top env-var seam above) -- the session-wide value is harmless
    # (it points under _SESSION_DATA_DIR, never the real root), but each test
    # still gets its own tmp_path here so per-test isolation is complete
    # rather than merely session-wide.
    monkeypatch.setattr(settings.audio, "temp_dir", tmp_path / "tmp")
    user_settings._settings = None
    with history._lock:
        history._close_conn_locked()
        history._output_dir = None  # AC 8a: lazy fallback inside history_path()
        history._stats_cache = None
    yield
    with history._lock:
        history._close_conn_locked()


def _monitored_app_data_paths() -> list[tuple[str, "Path"]]:
    """(label, current value) for every AC 5a-monitored path. A function, not
    a constant, and re-resolved on every call -- each of these five is
    itself either lazy or per-test-repointed, so a snapshot taken once would
    go stale immediately."""
    from app.core import logging_config
    from app.stt import local_setup

    getters: list[tuple[str, Callable[[], Path]]] = [
        ("history.history_path()", history.history_path),
        ("user_settings._settings_path()", user_settings._settings_path),
        ("settings.audio.temp_dir", lambda: settings.audio.temp_dir),
        ("logging_config.log_file_path()", logging_config.log_file_path),
        ("local_setup._crash_guard_path()", local_setup._crash_guard_path),
    ]
    resolved: list[tuple[str, Path]] = []
    for label, getter in getters:
        try:
            resolved.append((label, getter()))
        except Exception:
            # A test that monkeypatches a low-level platform primitive
            # (os.name, sys.platform) for an unrelated purpose -- e.g.
            # simulating a non-Windows GPU-vendor path in
            # test_stt_local_factory.py/test_gpu_probe.py -- can transiently
            # make pathlib's own Path() construction raise here (it picks
            # WindowsPath/PosixPath from the live os.name). That's a
            # collision with unrelated test state, not a data-isolation
            # leak -- skip a label we structurally can't evaluate rather
            # than crashing this guard fixture over it.
            continue
    return resolved


def _paths_under_real_root(real_roots) -> list[tuple[str, Path]]:
    """Return (label, path) pairs, from the AC 5a-monitored set, that
    currently resolve under any of `real_roots`. Empty when isolation is
    intact.

    `real_roots`: an iterable of root Paths to check against (this project
    has two -- prod and dev). Factored out of the guard fixture below so
    test_data_isolation.py can exercise the detection logic directly and
    prove it actually flags a leak, not just that it always happens to pass.
    """
    real_roots = list(real_roots)
    offenders: list[tuple[str, Path]] = []
    for label, path in _monitored_app_data_paths():
        for real_root in real_roots:
            try:
                is_real = path == real_root or path.is_relative_to(real_root)
            except ValueError:
                is_real = False
            if is_real:
                offenders.append((label, path))
                break
    return offenders


@pytest.fixture(autouse=True)
def _guard_against_real_app_data_root(_isolated_app_data, _real_app_data_roots):
    """Belt-and-braces net: fails loudly, with the offending path, if ANY of
    the five AC 5a-monitored paths ever points at the developer's REAL
    app-data root (prod or dev) instead of the test's tmp_path.

    Checked both before and after the test body. `_isolated_app_data`'s own
    reset already guarantees the pre-check passes trivially by construction
    (it just ran) -- the check that actually has teeth is the post-check: it
    catches pollution introduced *during* the test itself (e.g. a
    `TestClient(app)` lifespan re-pointing history at the real directory --
    the exact Spec 028 Item 1 bug) in the SAME test that caused it, rather
    than relying on a subsequent test's pre-check to notice (which it never
    would, since the next test's own `_isolated_app_data` reset runs first
    and would silently fix the leftover pollution before any check saw it).
    """
    offenders = _paths_under_real_root(_real_app_data_roots)
    assert not offenders, (
        f"Test isolation leak BEFORE the test body ran: {offenders} resolved "
        f"under a REAL app-data root {_real_app_data_roots} instead of "
        f"this test's tmp_path"
    )
    yield
    offenders = _paths_under_real_root(_real_app_data_roots)
    assert not offenders, (
        f"Test isolation leak AFTER the test body ran: {offenders} resolved "
        f"under a REAL app-data root {_real_app_data_roots} instead of "
        f"this test's tmp_path -- this test repointed a monitored path "
        f"at real user data"
    )


def _snapshot_real_roots(real_roots) -> dict:
    """AC 5c behavioural backstop: (exists, size, mtime) for each real root
    directory itself, plus every file ANYWHERE beneath it -- recursive, not
    just direct children. Needs no enumeration of consumers at all -- it is
    the net under AC 5a/5b's net.

    Recursive matters specifically because the two subdirectories actually
    written to in production -- `logs/` (backend.log) and `tmp/` (pipeline
    audio temp files) -- are not direct children the old `iterdir()`-based
    walk would see; review pass 2 caught this exact gap (a file written into
    `tmp/` went unnoticed) and proved it with a planted file.

    Uses `os.walk(..., onerror=...)` rather than `Path.rglob()`: a listing
    error partway through the walk (a directory removed mid-scan by a
    concurrent sibling agent on this shared machine, or ordinary `tmp/`
    churn from a real pipeline run) must not abort the whole snapshot --
    `onerror` swallows it and the walk continues with whatever it can still
    see. Each file's own `.stat()` call is separately guarded: a file that
    existed at listing time but vanished before `.stat()` runs (the same
    kind of transient churn) is skipped, not treated as a leak signal.
    """
    snapshot: dict = {}
    for root in real_roots:
        snapshot[root] = root.is_dir()
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None):
            for name in filenames:
                child = Path(dirpath) / name
                try:
                    st = child.stat()
                except OSError:
                    continue
                snapshot[child] = (st.st_size, st.st_mtime)
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def _snapshot_real_roots_backstop(_real_app_data_roots):
    """Session-scoped: records both real roots' state before the first test
    and asserts nothing changed after the last one. Attribution to a
    specific test is poor by design -- this is the net under AC 5a/5b's net,
    catching a leak through a mechanism nobody enumerated."""
    before = _snapshot_real_roots(_real_app_data_roots)
    yield
    after = _snapshot_real_roots(_real_app_data_roots)
    assert after == before, (
        "A real app-data root changed during the test session -- the suite "
        "wrote somewhere real.\n"
        f"before-only: {sorted(str(p) for p in before.keys() - after.keys())}\n"
        f"after-only (NEW): {sorted(str(p) for p in after.keys() - before.keys())}\n"
        f"changed: {sorted(str(p) for p in before.keys() & after.keys() if before[p] != after[p])}"
    )


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset settings and provider caches to defaults after each test.

    Also busts `gpu_probe`'s process-lifetime cache (added as part of the
    Spec 018 GitHub-review follow-up fix) so a test that exercises the real,
    unmocked `probe_gpu()` (e.g. `test_gpu_probe.py`, or a test that
    deliberately restores the real `get_local_provider_kind()` path) never
    leaks a cached result into the next test.
    """
    original_stt_mode = settings.stt.mode
    original_llm_mode = settings.llm.mode
    yield
    settings.stt.mode = original_stt_mode
    settings.llm.mode = original_llm_mode
    clear_stt_cache()
    clear_gpu_probe_cache()


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Clear app.dependency_overrides after every test so an override set in
    one test can never leak into the next."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _force_faster_whisper_for_local(monkeypatch, request):
    """Pin the local STT provider class to `LocalSTTProvider` for tests that
    are not specifically exercising the MLX path.

    On macOS Apple Silicon `get_local_provider_class()` returns
    `MLXWhisperSTTProvider`, which would break `isinstance(p, LocalSTTProvider)`
    assertions in `test_stt.py`, `test_stt_routing.py`, and `test_factories.py`.
    Patching the factory keeps those tests platform-agnostic. Tests that need
    the MLX path opt out via `@pytest.mark.mlx`.

    Also pins `get_local_provider_kind()` (spec 018) to `FASTER_WHISPER`:
    `local_setup.py`'s readiness-check functions (`_check_package_installed`,
    `ensure_local_ready`, `check_status`, `_estimate_model_ram_mb`) now call
    it directly, not only through `get_local_provider_class()`. This
    project's own dev machine has a real AMD GPU (spec 018) — the unpatched
    function would route those calls to `WHISPER_CPP_VULKAN` on THIS
    machine specifically, breaking the platform-agnostic guarantee this
    fixture already exists to provide. The stub accepts (and ignores) an
    optional positional `vendor` arg — `check_status()` (GitHub review on PR
    #21, iteration 1, issue #2) now calls the real function with an
    already-resolved vendor to avoid double-probing the GPU, and this stub
    must accept that same call shape.

    Patched on `app.stt.local_setup`'s own already-bound name (mirroring
    `is_macos_arm64`'s existing import style), NOT on `app.stt.local_factory`
    directly — `test_stt_local_factory.py`'s
    `test_factory_module_imports_no_third_party_at_module_level` deletes and
    re-imports `app.stt.local_factory` from `sys.modules`, which would
    silently split the patched module object from the one `local_setup.py`
    already imported its name from, un-patching this fixture for every test
    that runs after that one in the same session.

    Also opts out via `@pytest.mark.no_factory_stub` (spec 028 iteration-2
    review, RED 2): this fixture stubs `get_local_provider_class` itself, so
    any test asserting on the *cost* of calling it (e.g. "a Cloud request
    must trigger zero calls to the local factory / GPU probe") is worthless
    while this fixture is active -- it stubs the exact call under test,
    which is why the 574-test suite was structurally blind to the
    ~126 ms-per-Cloud-request probe regression ADR 018 documents.
    """
    if request.node.get_closest_marker("mlx") or request.node.get_closest_marker(
        "no_factory_stub"
    ):
        return
    from app.stt import local
    from app.stt.local_factory import LocalProviderKind
    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: local.LocalSTTProvider,
    )
    monkeypatch.setattr(
        "app.stt.local_setup.get_local_provider_kind",
        lambda *args, **kwargs: LocalProviderKind.FASTER_WHISPER,
    )


@pytest.fixture(autouse=True)
def _no_prewarm_by_default(monkeypatch, request):
    """No-op `maybe_prewarm_local`/`maybe_prewarm_local_at_startup` for every
    test except those marked `@pytest.mark.prewarm` — same opt-out shape as
    `_force_faster_whisper_for_local`.

    Without this, any test that flips STT mode to "local" (e.g. the
    pre-existing `test_set_stt_mode_accepts_json_object`) would spawn a real
    background pip-install/model-download task during the suite. Also resets
    the module-level `_prewarm_error` latch after every test so a failure
    injected by one test can't leak into the next.

    `maybe_prewarm_local_at_startup` (Spec 023) is no-op'd too so a future
    test exercising `TestClient(app)`'s real `lifespan()` with Local mode
    already active at boot can't accidentally read/write the real on-disk
    crash-guard marker or spawn a real background load during the suite.

    Also resets `_active_load` (Stage 5 GitHub review, PR #34, finding 1)
    after every test, suite-wide -- not just within test_stt_local_setup.py,
    since `test_pipeline.py`'s readiness-barrier tests exercise the real
    `ensure_local_ready`/`await_local_ready` too. It holds an `asyncio.Task`
    bound to the test's own event loop; left stale, a later test could try
    to `asyncio.shield()` a task from an already-closed loop.

    Also clears `app.core.tasks._background_tasks` (Spec 032, AC 14) for the
    same reason -- a strong-referenced `asyncio.Task` bound to a test's own
    (now-closed) event loop must not survive into the next test.
    """
    from app.core import tasks
    from app.stt import local_setup

    if not request.node.get_closest_marker("prewarm"):
        monkeypatch.setattr(local_setup, "maybe_prewarm_local", lambda stt_settings: None)
        monkeypatch.setattr(
            local_setup, "maybe_prewarm_local_at_startup", lambda stt_settings: None
        )
    yield
    local_setup._prewarm_error = None
    local_setup._active_load = None
    tasks._background_tasks.clear()


@pytest.fixture(autouse=True)
def _no_background_indexer_by_default(monkeypatch, request):
    """No-op `vector_store.run_background_indexer` for every test except
    those marked `@pytest.mark.background_indexer` -- same opt-out shape as
    `_no_prewarm_by_default`.

    Without this, any test using `TestClient(app)` (whose context manager
    runs the real FastAPI `lifespan()`) schedules a real background-indexing
    sweep against whatever `~/.justsay/history.db` and API keys exist on the
    machine running the suite -- see ADR 010 / spec 017 review RED #1.
    """
    from app.core import vector_store

    async def _noop() -> None:
        return None

    if not request.node.get_closest_marker("background_indexer"):
        monkeypatch.setattr(vector_store, "run_background_indexer", _noop)
