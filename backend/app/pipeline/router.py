"""Pipeline endpoints — unified audio-to-text flows."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from app.audio import MicrophoneRecorder, get_recorder
from app.core.config import settings
from app.core.constants import MAX_UPLOAD_SIZE
from app.core.utils import read_upload_with_limit
from app.pipeline.service import process_audio
from app.pipeline.upload_validation import validate_audio_upload

log = logging.getLogger(__name__)
router = APIRouter()


def _discard_scratch_file(path: Path) -> None:
    """Delete a scratch file without letting the delete replace the response.

    Both call sites sit in a ``finally``, where an ``OSError`` would throw away
    an already-built response: a completed transcription — copied to the
    clipboard and saved to history — would reach the widget as a bare 500.
    ``preferences/router.py``'s cleanup endpoint already guards the same
    operation in the same directory.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove scratch file %s", path, exc_info=True)


class DictateResponse(BaseModel):
    """Wire shape for /pipeline/dictate and /pipeline/process-file responses."""
    text: str
    duration_ms: int
    copied_to_clipboard: bool
    model_name: str = ""
    fallback_reason: str | None = None
    discarded_reason: str | None = None


@router.post("/dictate", response_model=DictateResponse)
async def dictate(
    background_tasks: BackgroundTasks,
    language: str = "uk",
    style: str = "normal",
    copy_to_clipboard: bool = True,
    recorder: MicrophoneRecorder = Depends(get_recorder),
):
    """One-shot: stop recording -> transcribe -> clipboard.

    Call POST /audio/start first, then call this endpoint when done speaking.
    The recorder reports the captured duration via ``last_duration_seconds`` so
    the pipeline can route short audio to Groq without re-reading the WAV.
    """
    if not recorder.is_recording:
        raise HTTPException(status_code=409, detail="Not recording. Call POST /audio/start first")

    audio_path = await recorder.stop()
    captured_duration = recorder.last_duration_seconds
    log.info(
        "Dictate: stopped recording. path=%s duration=%.2fs language=%s style=%s",
        audio_path.name, captured_duration, language, style,
    )

    try:
        result = await process_audio(
            audio_path,
            language=language,
            style=style,
            copy_to_clipboard=copy_to_clipboard,
            audio_duration=captured_duration if captured_duration > 0.0 else None,
            background_tasks=background_tasks,
        )
        return DictateResponse(**result.__dict__)
    except Exception as e:
        log.exception("Pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {type(e).__name__}: {e}",
        )
    finally:
        _discard_scratch_file(audio_path)


@router.post("/process-file", response_model=DictateResponse)
async def process_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    language: str = "auto",
    style: str = "normal",
    copy_to_clipboard: bool = True,
):
    """Process an uploaded audio file through the full pipeline."""
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    content = await read_upload_with_limit(file, MAX_UPLOAD_SIZE)
    validate_audio_upload(content, file.filename)

    temp_path = settings.audio.temp_dir / f"pipeline_{uuid.uuid4().hex}{ext}"

    try:
        settings.audio.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(content)

        result = await process_audio(
            temp_path,
            language=language,
            style=style,
            copy_to_clipboard=copy_to_clipboard,
            background_tasks=background_tasks,
        )
        return DictateResponse(**result.__dict__)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {type(e).__name__}: {e}",
        )
    finally:
        _discard_scratch_file(temp_path)
