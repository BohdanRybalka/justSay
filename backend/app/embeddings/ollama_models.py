"""Ollama model-listing helpers used by the local embedding availability probe.

``list_models``, ``model_matches`` and ``OllamaModel`` are consumed by
``app.embeddings.local.is_model_available`` to check whether Ollama reports a
given model pulled (ADR 029).
"""

import logging

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 5.0


class OllamaModel(BaseModel):
    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None


async def list_models(
    client: httpx.AsyncClient, target_model: str
) -> tuple[list[OllamaModel], bool, int | None]:
    """GET /api/tags -> {"models": [{"name": "gemma3:4b", "size": ..., "details": {...}}]}.

    Returns (all_models, target_is_downloaded, target_size_bytes).
    """
    try:
        r = await client.get("/api/tags")
        if r.status_code != 200:
            return [], False, None
        data = r.json()
    except Exception as e:
        log.warning("Ollama list models failed: %s", e)
        return [], False, None

    models: list[OllamaModel] = []
    downloaded = False
    target_size: int | None = None

    for m in data.get("models", []):
        name = m.get("name", "")
        size = m.get("size")
        details = m.get("details") or {}
        parameter_size = details.get("parameter_size")
        models.append(OllamaModel(name=name, size_bytes=size, parameter_size=parameter_size))
        if model_matches(name, target_model):
            downloaded = True
            target_size = size

    return models, downloaded, target_size


def model_matches(actual: str, target: str) -> bool:
    """Compare Ollama model names, tolerating a missing tag on either side.

    An exact match wins; otherwise a side carrying no ``:`` tag matches the other
    when that other one starts with it plus ``:``.
    """
    if actual == target:
        return True
    if ":" not in target and actual.startswith(f"{target}:"):
        return True
    if ":" not in actual and target.startswith(f"{actual}:"):
        return True
    return False
