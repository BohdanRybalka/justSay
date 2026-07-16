from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.audio import MicrophoneRecorder, get_recorder


router = APIRouter()


class RecordingStatus(BaseModel):
    is_recording: bool
    duration_seconds: float
    level_db: float


class StopResponse(BaseModel):
    filename: str
    duration_seconds: float


@router.post("/start", response_model=RecordingStatus)
async def start_recording(recorder: MicrophoneRecorder = Depends(get_recorder)):
    if recorder.is_recording:
        raise HTTPException(status_code=409, detail="Already recording")
    await recorder.start()
    return RecordingStatus(
        is_recording=recorder.is_recording,
        duration_seconds=recorder.duration_seconds,
        level_db=recorder.level_db,
    )


@router.post("/stop", response_model=StopResponse)
async def stop_recording(recorder: MicrophoneRecorder = Depends(get_recorder)):
    if not recorder.is_recording:
        raise HTTPException(status_code=409, detail="Not recording")
    duration = recorder.duration_seconds
    audio_path = await recorder.stop()
    return StopResponse(
        filename=audio_path.name,
        duration_seconds=duration,
    )


@router.get("/status", response_model=RecordingStatus)
async def recording_status(recorder: MicrophoneRecorder = Depends(get_recorder)):
    return RecordingStatus(
        is_recording=recorder.is_recording,
        duration_seconds=recorder.duration_seconds,
        level_db=recorder.level_db,
    )
