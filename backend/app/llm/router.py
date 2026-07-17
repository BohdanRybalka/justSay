from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.constants import MAX_TEXT_LENGTH
from app.core.types import ProviderMode
from app.core.user_settings import update_user_settings
from app.llm import get_llm_provider, clear_cache
from app.llm.local_setup import (
    LocalLlmStatus,
    check_status as check_local_status,
    load_ollama_model,
    pull_model,
    start_ollama,
    unload_ollama_model,
)
from app.llm.tasks import DEFAULT_TASK, TASK_PROFILES

router = APIRouter()

# Business-facing `style` vocabulary -> internal task name (an `app.llm.tasks
# .TASK_PROFILES` key). TASK_PROFILES is the single source of truth for valid task
# names; the values below are checked against it immediately so a future
# TASK_PROFILES key rename that isn't mirrored here fails at import time instead of
# this map silently falling back to DEFAULT_TASK on every affected request.
STYLE_TASK_MAP: dict[str, str] = {
    "normal": DEFAULT_TASK,
    "ai_prompt": "ai_prompt_structuring",
}

_unmapped_tasks = set(STYLE_TASK_MAP.values()) - set(TASK_PROFILES)
if _unmapped_tasks:
    raise RuntimeError(
        f"STYLE_TASK_MAP references task(s) not present in TASK_PROFILES: "
        f"{sorted(_unmapped_tasks)}"
    )


class ProcessRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    system_prompt: str = Field(..., max_length=MAX_TEXT_LENGTH)
    style: str = "normal"


class ProcessResponse(BaseModel):
    result: str
    model: str


class _ModeBody(BaseModel):
    """Inline body wrapper so the wire format stays ``{"mode": "..."}``."""
    mode: ProviderMode


@router.put("/mode")
async def set_llm_mode(body: _ModeBody):
    settings.llm.mode = body.mode
    clear_cache()
    update_user_settings({"llm_mode": body.mode.value})
    provider = get_llm_provider(settings.llm)
    return {"llm_mode": settings.llm.mode, "model": provider.model_name}


@router.post("/process", response_model=ProcessResponse)
async def process_text(req: ProcessRequest):
    """Process text with the current LLM provider."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        provider = get_llm_provider(settings.llm)
        task = STYLE_TASK_MAP.get(req.style, DEFAULT_TASK)
        result = await provider.process(req.text, req.system_prompt, task=task)
        return ProcessResponse(result=result, model=provider.model_name)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {type(e).__name__}"
        )


@router.get("/local/status", response_model=LocalLlmStatus)
async def llm_local_status():
    """Check local LLM readiness: Ollama running, model pulled, VRAM usage."""
    return await check_local_status(settings.llm)


@router.post("/local/pull")
async def llm_local_pull():
    """Pull Ollama model with streaming progress via SSE."""
    return StreamingResponse(
        pull_model(settings.llm),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/local/start")
async def llm_local_start():
    """Attempt to start Ollama serve process if host is local."""
    return await start_ollama(settings.llm)


@router.post("/local/load")
async def llm_local_load():
    """Prime the Ollama model in memory (warm-up with empty prompt)."""
    return await load_ollama_model(settings.llm)


@router.post("/local/unload")
async def llm_local_unload():
    """Release Ollama VRAM by setting keep_alive=0."""
    return await unload_ollama_model(settings.llm)
