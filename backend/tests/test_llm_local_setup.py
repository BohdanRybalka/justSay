import httpx
import pytest

from app.llm.local_setup import _list_models, _model_matches


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


# --- _list_models ---


_BASE = "http://localhost:11434"


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
