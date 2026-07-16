import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
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
    """Reset settings and provider caches to defaults after each test."""
    original_stt_mode = settings.stt.mode
    original_llm_mode = settings.llm.mode
    yield
    settings.stt.mode = original_stt_mode
    settings.llm.mode = original_llm_mode
    clear_stt_cache()
    clear_llm_cache()


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
    """
    if request.node.get_closest_marker("mlx"):
        return
    from app.stt import local
    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: local.LocalSTTProvider,
    )
