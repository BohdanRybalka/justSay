"""LLM Local mode readiness checks — Ollama health, model availability, VRAM, pull, start."""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from app.llm.config import LLMSettings

log = logging.getLogger(__name__)

_pull_lock = asyncio.Lock()

_CONNECT_TIMEOUT = 3.0
_READ_TIMEOUT = 5.0

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


class OllamaModel(BaseModel):
    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None


class LocalLlmStatus(BaseModel):
    ollama_running: bool = False
    ollama_version: str | None = None
    model_downloaded: bool = False
    model_name: str = ""
    model_size_bytes: int | None = None
    model_loaded: bool = False
    vram_used_bytes: int | None = None
    available_models: list[OllamaModel] = []


async def check_status(llm_settings: LLMSettings) -> LocalLlmStatus:
    """Aggregate Ollama health, version, model availability and load state."""
    target_model = llm_settings.ollama_model
    status = LocalLlmStatus(model_name=target_model)

    timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT)
    async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
        status.ollama_running = await _check_health(client)
        if not status.ollama_running:
            return status

        status.ollama_version = await _get_version(client)
        available, downloaded, size_bytes = await _list_models(client, target_model)
        status.available_models = available
        status.model_downloaded = downloaded
        status.model_size_bytes = size_bytes

        if downloaded:
            loaded, vram = await _check_loaded(client, target_model)
            status.model_loaded = loaded
            status.vram_used_bytes = vram

    return status


async def _check_health(client: httpx.AsyncClient) -> bool:
    """Ollama responds with 200 OK on GET /."""
    try:
        r = await client.get("/")
        return r.status_code == 200
    except Exception:
        return False


async def _get_version(client: httpx.AsyncClient) -> str | None:
    """GET /api/version -> {"version": "0.9.2"}"""
    try:
        r = await client.get("/api/version")
        if r.status_code == 200:
            return r.json().get("version")
    except Exception as e:
        log.warning("Ollama version check failed: %s", e)
    return None


async def _list_models(
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
        if _model_matches(name, target_model):
            downloaded = True
            target_size = size

    return models, downloaded, target_size


async def _check_loaded(client: httpx.AsyncClient, model: str) -> tuple[bool, int | None]:
    """GET /api/ps -> list of loaded models with size_vram.

    Returns (is_loaded, vram_bytes_or_none).
    """
    try:
        r = await client.get("/api/ps")
        if r.status_code != 200:
            return False, None
        data = r.json()
    except Exception as e:
        log.warning("Ollama ps check failed: %s", e)
        return False, None

    for m in data.get("models", []):
        if _model_matches(m.get("name", ""), model):
            vram = m.get("size_vram")
            return True, vram

    return False, None


async def pull_model(llm_settings: LLMSettings) -> AsyncIterator[str]:
    """Proxy POST /api/pull with stream=true, re-emitting NDJSON as SSE events.

    Yields SSE-formatted strings. Caller wraps in StreamingResponse(media_type='text/event-stream').
    """
    if _pull_lock.locked():
        yield _sse("error", {"status": "error", "error": "Pull already in progress"})
        return

    async with _pull_lock:
        # Pre-flight: Ollama must be reachable.
        preflight_timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT)
        try:
            async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=preflight_timeout) as client:
                if not await _check_health(client):
                    yield _sse("error", {"status": "error", "error": "Ollama not running"})
                    return
        except Exception:
            yield _sse("error", {"status": "error", "error": "Ollama not running"})
            return

        timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=None, write=None, pool=None)
        payload = {"name": llm_settings.ollama_model, "stream": True}

        try:
            async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
                async with client.stream("POST", "/api/pull", json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield _sse("error", {"status": "error", "error": f"HTTP {resp.status_code}: {body.decode('utf-8', errors='replace')[:300]}"})
                        return

                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in data:
                            yield _sse("error", {"status": "error", "error": data["error"]})
                            return

                        if data.get("status") == "success":
                            yield _sse("done", {"status": "success"})
                            return

                        yield _sse("progress", data)
        except httpx.ConnectError as e:
            yield _sse("error", {"status": "error", "error": f"Cannot reach Ollama at {llm_settings.ollama_host}: {e}"})
        except Exception as e:
            log.warning("Ollama pull failed: %s", e)
            yield _sse("error", {"status": "error", "error": str(e)})


def _is_local_host(host_url: str) -> bool:
    """True if host_url points at this machine (safe to auto-start Ollama)."""
    try:
        parsed = httpx.URL(host_url)
        return (parsed.host or "").lower() in _LOCALHOST_HOSTS
    except Exception:
        return False


async def start_ollama(llm_settings: LLMSettings) -> dict:
    """Spawn `ollama serve` in background and wait for /  to respond.

    Returns {started, error}. Only attempts spawn when host is local.
    """
    # Already running? Check first so we can report success even for remote hosts.
    timeout = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=2.0)
    try:
        async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
            if await _check_health(client):
                return {"started": True, "error": None}
    except Exception:
        pass

    if not _is_local_host(llm_settings.ollama_host):
        return {"started": False, "error": f"Cannot auto-start remote Ollama at {llm_settings.ollama_host}"}

    if shutil.which("ollama") is None:
        return {"started": False, "error": "ollama executable not found in PATH"}

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as e:
        log.warning("Failed to spawn 'ollama serve': %s", e)
        return {"started": False, "error": f"Failed to spawn ollama: {e}"}

    # Poll for readiness up to 10s.
    deadline_attempts = 20
    for _ in range(deadline_attempts):
        await asyncio.sleep(0.5)
        try:
            async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
                if await _check_health(client):
                    return {"started": True, "error": None}
        except Exception:
            continue

    return {"started": False, "error": "Ollama spawn timed out"}


async def load_ollama_model(llm_settings: LLMSettings) -> dict:
    """Prime the target model in Ollama's memory by sending an empty generate call.

    Ollama loads the model on first request; a zero-prompt request warms it up.
    Returns {loaded, error}.
    """
    timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=None, write=None, pool=None)
    payload = {
        "model": llm_settings.ollama_model,
        "prompt": "",
        "stream": False,
        "keep_alive": "5m",
    }
    try:
        async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
            r = await client.post("/api/generate", json=payload)
            if r.status_code != 200:
                return {"loaded": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        return {"loaded": True, "error": None}
    except Exception as e:
        log.warning("Ollama load failed: %s", e)
        return {"loaded": False, "error": str(e)}


async def unload_ollama_model(llm_settings: LLMSettings) -> dict:
    """Ask Ollama to release VRAM by setting keep_alive=0 on a noop generate.

    Returns {unloaded, error}.
    """
    timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT)
    payload = {
        "model": llm_settings.ollama_model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    try:
        async with httpx.AsyncClient(base_url=llm_settings.ollama_host, timeout=timeout) as client:
            r = await client.post("/api/generate", json=payload)
            if r.status_code != 200:
                return {"unloaded": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        return {"unloaded": True, "error": None}
    except Exception as e:
        log.warning("Ollama unload failed: %s", e)
        return {"unloaded": False, "error": str(e)}


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _model_matches(actual: str, target: str) -> bool:
    """Compare Ollama model names tolerating a missing tag on either side.

    Rules:
      - Exact match wins.
      - If target has no ':' tag, any actual that starts with 'target:' matches
        (e.g. target='gemma3' matches actual 'gemma3:4b' and 'gemma3:latest').
      - Symmetrically, if actual has no ':' tag, it matches 'actual:<any>' target.
    """
    if actual == target:
        return True
    if ":" not in target and actual.startswith(f"{target}:"):
        return True
    if ":" not in actual and target.startswith(f"{actual}:"):
        return True
    return False
