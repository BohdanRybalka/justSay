import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.audio_validation import validate_audio_upload
from app.core.config import settings
from app.core.constants import MAX_UPLOAD_SIZE
from app.core.types import ProviderMode
from app.core.user_settings import update_user_settings
from app.core.utils import read_upload_with_limit
from app.stt import clear_cache, get_provider
from app.stt.local_setup import (
    LocalSttStatus,
    check_status as check_local_status,
    install_local_packages,
)

router = APIRouter()


class TranscribeResponse(BaseModel):
    text: str
    model: str


class _ModeBody(BaseModel):
    """Inline body wrapper so the wire format stays ``{"mode": "..."}``."""
    mode: ProviderMode


@router.put("/mode")
async def set_stt_mode(body: _ModeBody):
    settings.stt.mode = body.mode
    clear_cache()
    update_user_settings({"stt_mode": body.mode.value})
    provider = get_provider(settings.stt.mode, settings.stt)
    # Lazy import (not a module-level from-import) so the autouse conftest
    # fixture — and tests that monkeypatch app.stt.local_setup directly —
    # can intercept this call; same pattern as app.stt._get_local's factory
    # patch comment.
    from app.stt.local_setup import maybe_prewarm_local

    # Fire-and-forget: no-op for Cloud, kicks off an eager pre-warm task for
    # Local without making this response wait on the model load.
    maybe_prewarm_local(settings.stt)
    return {"stt_mode": settings.stt.mode, "model": provider.model_name}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile, language: str = "uk"):
    """Transcribe an uploaded audio file using the current STT provider.

    Deliberately NOT covered by the pipeline's silence gate
    (`app.pipeline.service.process_audio`'s neural VAD, with `analyze_silence`
    as the lazy fallback when the VAD abstains) -- this is a raw
    provider-passthrough dev/test surface with no frontend caller, and a gate
    here would mask the very provider behaviour someone would be using this
    endpoint to observe. See docs/adr/015-pipeline-level-silence-guard.md and
    020-lazy-energy-guard-fallback.md.
    """
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    # Read first so we can validate magic bytes, not just the filename.
    content = await read_upload_with_limit(file, MAX_UPLOAD_SIZE)
    # Raises HTTPException(400) on bad extension / bad magic / mismatch.
    validate_audio_upload(content, file.filename)

    temp_path = settings.audio.temp_dir / f"upload_{uuid.uuid4().hex}{ext}"

    try:
        settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(content)

        provider = get_provider(settings.stt.mode, settings.stt)
        result = await provider.transcribe(temp_path, language=language)

        return TranscribeResponse(text=result.text, model=provider.model_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {type(e).__name__}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.get("/local/status", response_model=LocalSttStatus)
async def stt_local_status():
    """Check local STT readiness: package, model loaded, GPU."""
    return await asyncio.to_thread(check_local_status, settings.stt)


@router.post("/local/load")
async def stt_local_load():
    """Load whisper model into memory. May take minutes on first run (model download)."""
    if settings.stt.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")

    try:
        provider = get_provider(settings.stt.mode, settings.stt)
        # _get_model() triggers lazy load — run in thread (blocking, downloads model)
        await asyncio.to_thread(provider._get_model)
        return {"loaded": True, "model": provider.model_name}
    except Exception as e:
        # _get_model already latched the error message. Surface it to the user.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/local/unload")
async def stt_local_unload():
    """Unload whisper model from memory, free GPU/RAM."""
    clear_cache()
    return {"unloaded": True}


@router.post("/local/prewarm")
async def stt_local_prewarm():
    """Retry affordance for the Local STT status indicator's error state.

    Fire-and-forget, same as the automatic pre-warm triggers — returns
    immediately, does not await the install/load itself.
    """
    if settings.stt.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")
    from app.stt.local_setup import maybe_prewarm_local

    maybe_prewarm_local(settings.stt)
    return {"started": True}


@router.post("/local/install")
async def stt_local_install():
    """Install local STT dependencies (pip install .[local]) with SSE progress."""
    return StreamingResponse(
        install_local_packages(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
