"""Pipeline endpoints — unified audio-to-text flows."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.audio.dependencies import get_recorder
from app.audio.recorder import MicrophoneRecorder
from app.audio.session import SessionRef
from app.core.errors import JustSayError
from app.core.scratch import discard_scratch_file
from app.pipeline.jobs import dictation_gate
from app.pipeline.service import process_audio

log = logging.getLogger(__name__)
router = APIRouter()

_PIPELINE_CRASHED_DETAIL = (
    "The transcription pipeline failed unexpectedly. Check the backend log."
)


class DictateResponse(BaseModel):
    """Wire shape for /pipeline/dictate responses."""
    text: str
    duration_ms: int
    copied_to_clipboard: bool
    model_name: str = ""
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
        with dictation_gate.dictating():
            result = await process_audio(
                audio_path,
                language=language,
                copy_to_clipboard=copy_to_clipboard,
                audio_duration=captured_duration if captured_duration > 0.0 else None,
                background_tasks=background_tasks,
                source="dictation",
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
        discard_scratch_file(audio_path)
