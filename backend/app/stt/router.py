import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.schemas import ProviderModeUpdate
from app.core.user_settings import update_user_settings
from app.stt import get_stt_provider, clear_cache
from app.stt.local_setup import (
    LocalSttStatus,
    check_status as check_local_status,
    install_local_packages,
)

router = APIRouter()

MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac",
    ".m4a", ".mp4", ".aac", ".opus", ".wma", ".aiff", ".aif",
}


class TranscribeResponse(BaseModel):
    text: str
    model: str


@router.put("/mode")
async def set_stt_mode(update: ProviderModeUpdate):
    settings.stt.mode = update.mode
    clear_cache()
    update_user_settings({"stt_mode": update.mode.value})
    provider = get_stt_provider(settings.stt)
    return {"stt_mode": settings.stt.mode, "model": provider.model_name}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile, language: str = "uk"):
    """Transcribe an uploaded audio file using the current STT provider."""
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_path = settings.audio.temp_dir / f"upload_{uuid.uuid4().hex}{ext}"

    try:
        settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)
        content = await _read_with_limit(file, MAX_UPLOAD_SIZE)
        temp_path.write_bytes(content)

        provider = get_stt_provider(settings.stt)
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
    from app.core.types import ProviderMode

    if settings.stt.mode != ProviderMode.LOCAL:
        raise HTTPException(status_code=400, detail="STT mode is not local")

    try:
        provider = get_stt_provider(settings.stt)
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


@router.post("/local/install")
async def stt_local_install():
    """Install local STT dependencies (pip install .[local]) with SSE progress."""
    return StreamingResponse(
        install_local_packages(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _read_with_limit(file: UploadFile, max_size: int) -> bytes:
    """Read upload file in chunks, enforcing size limit."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_size // 1024 // 1024}MB)")
        chunks.append(chunk)
    return b"".join(chunks)
