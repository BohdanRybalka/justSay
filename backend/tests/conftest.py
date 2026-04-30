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
