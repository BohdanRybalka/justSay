from unittest.mock import AsyncMock, patch

import pytest
import httpx

from app.core.utils import sse_event
from app.llm.config import LLMSettings
from app.llm.local_setup import (
    check_status,
    pull_model,
    start_ollama,
    _check_health,
    _get_version,
    _list_models,
    _check_loaded,
    _model_matches,
    _is_local_host,
)


# --- _model_matches ---


def test_model_matches_exact():
    assert _model_matches("gemma3:4b", "gemma3:4b") is True


def test_model_matches_no_tag_target():
    assert _model_matches("gemma3:4b", "gemma3") is True
    assert _model_matches("gemma3:latest", "gemma3") is True


def test_model_matches_different_tag():
    assert _model_matches("gemma3:latest", "gemma3:4b") is False


def test_model_matches_different_model():
    assert _model_matches("llama3:8b", "gemma3:4b") is False


# --- _check_health ---


_BASE = "http://localhost:11434"


@pytest.mark.asyncio
async def test_health_running():
    mock_response = httpx.Response(200, text="Ollama is running")
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        result = await _check_health(client)
    assert result is True


@pytest.mark.asyncio
async def test_health_not_running():
    def raise_connect_error(req):
        raise httpx.ConnectError("Connection refused")

    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(raise_connect_error)) as client:
        result = await _check_health(client)
    assert result is False


# --- _get_version ---


@pytest.mark.asyncio
async def test_get_version_ok():
    mock_response = httpx.Response(200, json={"version": "0.9.2"})
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        result = await _get_version(client)
    assert result == "0.9.2"


@pytest.mark.asyncio
async def test_get_version_error():
    mock_response = httpx.Response(500)
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        result = await _get_version(client)
    assert result is None


# --- _list_models ---


@pytest.mark.asyncio
async def test_list_models_target_found():
    mock_response = httpx.Response(200, json={
        "models": [
            {"name": "gemma3:4b", "size": 3_341_680_640, "details": {"parameter_size": "4B"}},
            {"name": "llama3:8b", "size": 4_000_000_000, "details": {"parameter_size": "8B"}},
        ]
    })
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        models, found, size = await _list_models(client, "gemma3:4b")
    assert len(models) == 2
    assert models[0].name == "gemma3:4b"
    assert models[0].parameter_size == "4B"
    assert found is True
    assert size == 3_341_680_640


@pytest.mark.asyncio
async def test_list_models_target_not_found():
    mock_response = httpx.Response(200, json={
        "models": [
            {"name": "llama3:8b", "size": 4_000_000_000, "details": {}},
        ]
    })
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        models, found, size = await _list_models(client, "gemma3:4b")
    assert len(models) == 1
    assert found is False
    assert size is None


@pytest.mark.asyncio
async def test_list_models_empty():
    mock_response = httpx.Response(200, json={"models": []})
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        models, found, size = await _list_models(client, "gemma3:4b")
    assert len(models) == 0
    assert found is False


# --- _check_loaded ---


@pytest.mark.asyncio
async def test_check_loaded_in_memory():
    mock_response = httpx.Response(200, json={
        "models": [
            {"name": "gemma3:4b", "size_vram": 1_800_000_000},
        ]
    })
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        loaded, vram = await _check_loaded(client, "gemma3:4b")
    assert loaded is True
    assert vram == 1_800_000_000


@pytest.mark.asyncio
async def test_check_loaded_not_in_memory():
    mock_response = httpx.Response(200, json={"models": []})
    async with httpx.AsyncClient(base_url=_BASE, transport=httpx.MockTransport(lambda req: mock_response)) as client:
        loaded, vram = await _check_loaded(client, "gemma3:4b")
    assert loaded is False
    assert vram is None


# --- check_status (integration) ---


@pytest.mark.asyncio
async def test_status_ollama_not_running():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    def raise_connect_error(req):
        raise httpx.ConnectError("Connection refused")

    with patch("app.llm.local_setup.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        status = await check_status(settings)

    assert status.ollama_running is False
    assert status.model_downloaded is False
    assert status.model_name == "gemma3:4b"


@pytest.mark.asyncio
async def test_status_all_ready():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    call_count = 0

    def mock_transport(req: httpx.Request) -> httpx.Response:
        nonlocal call_count
        path = req.url.path

        if path == "/":
            return httpx.Response(200, text="Ollama is running")
        elif path == "/api/version":
            return httpx.Response(200, json={"version": "0.9.2"})
        elif path == "/api/tags":
            return httpx.Response(200, json={
                "models": [{"name": "gemma3:4b", "size": 3_341_680_640}]
            })
        elif path == "/api/ps":
            return httpx.Response(200, json={
                "models": [{"name": "gemma3:4b", "size_vram": 1_800_000_000}]
            })
        return httpx.Response(404)

    with patch("app.llm.local_setup.httpx.AsyncClient", wraps=lambda **kw: httpx.AsyncClient(
        transport=httpx.MockTransport(mock_transport), **{k: v for k, v in kw.items() if k not in ("base_url", "timeout")}
    )) as _:
        # Use a more direct approach — patch the individual check functions
        pass

    # Simpler: patch the internal functions directly
    with (
        patch("app.llm.local_setup._check_health", return_value=True),
        patch("app.llm.local_setup._get_version", return_value="0.9.2"),
        patch("app.llm.local_setup._list_models", return_value=([], True, 3_341_680_640)),
        patch("app.llm.local_setup._check_loaded", return_value=(True, 1_800_000_000)),
    ):
        status = await check_status(settings)

    assert status.ollama_running is True
    assert status.ollama_version == "0.9.2"
    assert status.model_downloaded is True
    assert status.model_size_bytes == 3_341_680_640
    assert status.model_loaded is True
    assert status.vram_used_bytes == 1_800_000_000


@pytest.mark.asyncio
async def test_status_model_not_pulled():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    with (
        patch("app.llm.local_setup._check_health", return_value=True),
        patch("app.llm.local_setup._get_version", return_value="0.9.2"),
        patch("app.llm.local_setup._list_models", return_value=([], False, None)),
        patch("app.llm.local_setup._check_loaded", return_value=(False, None)),
    ):
        status = await check_status(settings)

    assert status.ollama_running is True
    assert status.model_downloaded is False
    assert status.model_size_bytes is None


# --- sse_event helper ---


def test_sse_format():
    result = sse_event("progress", {"status": "pulling", "completed": 100})
    assert result.startswith("event: progress\n")
    assert '"status": "pulling"' in result
    assert result.endswith("\n\n")


# --- pull_model ---


@pytest.mark.asyncio
async def test_pull_model_ollama_not_running():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    def raise_connect(req):
        raise httpx.ConnectError("Connection refused")

    with patch("app.llm.local_setup.httpx.AsyncClient", return_value=httpx.AsyncClient(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(raise_connect),
    )):
        events = []
        async for event in pull_model(settings):
            events.append(event)

    assert len(events) == 1
    assert "error" in events[0]
    assert "Ollama not running" in events[0]


@pytest.mark.asyncio
async def test_pull_model_already_in_progress():
    """Test that concurrent pulls are rejected."""
    import asyncio
    from app.llm.local_setup import _pull_lock

    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    # Simulate lock being held
    await _pull_lock.acquire()
    try:
        events = []
        async for event in pull_model(settings):
            events.append(event)

        assert len(events) == 1
        assert "Pull already in progress" in events[0]
    finally:
        _pull_lock.release()


# --- _is_local_host ---


def test_is_local_host_localhost():
    assert _is_local_host("http://localhost:11434") is True


def test_is_local_host_127():
    assert _is_local_host("http://127.0.0.1:11434") is True


def test_is_local_host_remote():
    assert _is_local_host("http://192.168.1.50:11434") is False


# --- start_ollama ---


@pytest.mark.asyncio
async def test_start_ollama_already_running():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    with patch("app.llm.local_setup.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_resp)

        result = await start_ollama(settings)

    assert result["started"] is True
    assert result["error"] is None


@pytest.mark.asyncio
async def test_start_ollama_not_in_path():
    settings = LLMSettings(ollama_host="http://localhost:11434", ollama_model="gemma3:4b")

    with (
        patch("app.llm.local_setup.httpx.AsyncClient") as MockClient,
        patch("app.llm.local_setup.shutil.which", return_value=None),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        result = await start_ollama(settings)

    assert result["started"] is False
    assert "not found in PATH" in result["error"]


@pytest.mark.asyncio
async def test_start_ollama_remote_host_not_reachable():
    settings = LLMSettings(ollama_host="http://192.168.1.50:11434", ollama_model="gemma3:4b")

    with patch("app.llm.local_setup.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        result = await start_ollama(settings)

    assert result["started"] is False
    assert "Cannot auto-start remote" in result["error"]
