import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.audio import MicrophoneRecorder, get_recorder
from app.core.utils import sse_event


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


async def _level_stream(request: Request, recorder: MicrophoneRecorder):
    while True:
        if await request.is_disconnected():
            return
        if not recorder.is_recording:
            yield sse_event("done", {"is_recording": False})
            return
        yield sse_event("level", {"level_db": recorder.level_db, "is_recording": True})
        await asyncio.sleep(0.1)


@router.get("/level-stream")
async def level_stream(request: Request, recorder: MicrophoneRecorder = Depends(get_recorder)):
    return StreamingResponse(
        _level_stream(request, recorder),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
