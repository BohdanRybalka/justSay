"""Pipeline endpoints — unified audio-to-text flows."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from app.audio.dependencies import get_recorder
from app.audio.recorder import MicrophoneRecorder
from app.audio.session import SessionRef
from app.core.config import settings
from app.core.constants import MAX_UPLOAD_SIZE
from app.core.errors import JustSayError
from app.pipeline.service import process_audio
from app.pipeline.upload_validation import read_upload_with_limit, validate_audio_upload

log = logging.getLogger(__name__)
router = APIRouter()

_PIPELINE_CRASHED_DETAIL = (
    "The transcription pipeline failed unexpectedly. Check the backend log."
)


def _discard_scratch_file(path: Path) -> None:
    """Delete a scratch file without letting the delete replace the response.

    Both call sites sit in a ``finally``, where an ``OSError`` would turn an
    already-built response into a bare 500.
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
    ref: SessionRef | None = None,
    language: str = "uk",
    copy_to_clipboard: bool = True,
    recorder: MicrophoneRecorder = Depends(get_recorder),
):
    """One-shot: stop recording -> transcribe -> clipboard.

    Call POST /audio/start first, then this when done speaking. An optional
    ``session_id`` names the recording meant; a stranger's is refused with 403.
    """
    if not recorder.is_recording:
        raise HTTPException(status_code=409, detail="Not recording. Call POST /audio/start first")

    audio_path = await recorder.stop(ref.session_id if ref else None)
    captured_duration = recorder.last_duration_seconds
    log.info(
        "Dictate: stopped recording. path=%s duration=%.2fs language=%s",
        audio_path.name, captured_duration, language,
    )

    try:
        result = await process_audio(
            audio_path,
            language=language,
            copy_to_clipboard=copy_to_clipboard,
            audio_duration=captured_duration if captured_duration > 0.0 else None,
            background_tasks=background_tasks,
        )
        return DictateResponse(**result.__dict__)
    except JustSayError:
        raise
    except Exception:
        log.exception("Pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=_PIPELINE_CRASHED_DETAIL,
        )
    finally:
        _discard_scratch_file(audio_path)


@router.post("/process-file", response_model=DictateResponse)
async def process_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    language: str = "auto",
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
            copy_to_clipboard=copy_to_clipboard,
            background_tasks=background_tasks,
        )
        return DictateResponse(**result.__dict__)
    except HTTPException:
        raise
    except JustSayError:
        raise
    except Exception:
        log.exception("Pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=_PIPELINE_CRASHED_DETAIL,
        )
    finally:
        _discard_scratch_file(temp_path)
