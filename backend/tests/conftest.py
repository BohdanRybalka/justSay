import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.gpu_probe import clear_cache as clear_gpu_probe_cache
from app.main import app
from app.stt import clear_cache as clear_stt_cache
from app.llm import clear_cache as clear_llm_cache


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


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
    clear_llm_cache()
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
    """
    if request.node.get_closest_marker("mlx"):
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
    """
    from app.stt import local_setup

    if not request.node.get_closest_marker("prewarm"):
        monkeypatch.setattr(local_setup, "maybe_prewarm_local", lambda stt_settings: None)
        monkeypatch.setattr(
            local_setup, "maybe_prewarm_local_at_startup", lambda stt_settings: None
        )
    yield
    local_setup._prewarm_error = None


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
