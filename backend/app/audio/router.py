import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.audio import (
    MeetingRecorder,
    MicrophoneRecorder,
    get_active_meeting_recorder,
    get_active_recorder,
    get_meeting_recorder,
    get_recorder,
)
from app.audio.meeting_recorder import MeetingCaptureAbortedError
from app.audio.system_source import SystemAudioUnavailableError
from app.core.utils import sse_event
from app.preferences.user_settings import get_user_settings

router = APIRouter()


class RecordingStatus(BaseModel):
    is_recording: bool
    duration_seconds: float
    level_db: float


class StopResponse(BaseModel):
    filename: str
    duration_seconds: float


class MeetingStopResponse(BaseModel):
    """Deliberately separate from StopResponse.

    The Instant Prompt response model must stay untouched by this feature, and
    the meeting path reports one thing dictation does not: whether the raw
    buffer cap was hit and audio was dropped.
    """

    filename: str
    duration_seconds: float
    truncated: bool


class MeetingStatus(BaseModel):
    """Separate from RecordingStatus for the reason MeetingStopResponse is.

    The dictation contract must not move, and the meeting path has to report
    which output it is capturing and whether sound is arriving from it.
    """

    is_recording: bool
    duration_seconds: float
    level_db: float
    system_endpoint: str | None
    system_level_db: float


_CONSENT_REQUIRED_DETAIL = (
    "Meeting recording has not been acknowledged — open Settings → General and "
    "confirm you are responsible for obtaining the participants' consent"
)


def _meeting_status(recorder: MeetingRecorder) -> MeetingStatus:
    return MeetingStatus(
        is_recording=recorder.is_recording,
        duration_seconds=recorder.duration_seconds,
        level_db=recorder.level_db,
        system_endpoint=recorder.system_endpoint,
        system_level_db=recorder.system_level_db,
    )


@router.post("/start", response_model=RecordingStatus)
async def start_recording(
    recorder: MicrophoneRecorder = Depends(get_recorder),
    meeting_recorder: MeetingRecorder | None = Depends(get_active_meeting_recorder),
):
    if meeting_recorder is not None and meeting_recorder.is_busy:
        raise HTTPException(status_code=409, detail="A meeting recording is in progress")
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


@router.post("/meeting/start", response_model=MeetingStatus)
async def start_meeting_recording(
    recorder: MeetingRecorder = Depends(get_meeting_recorder),
    dictation_recorder: MicrophoneRecorder | None = Depends(get_active_recorder),
):
    """Begin capturing the microphone and the system output together.

    Answers 403 until the meeting disclosure has been acknowledged, which is
    what makes the dialog impossible to drive around with curl — see
    docs/adr/040-recording-other-people-is-not-covered-by-zero-leak.md. A
    platform with no system-audio implementation answers 501 and opens no
    stream at all.

    The `is_busy` guard is a conservative filter, not a decision: it can only
    refuse, never permit something the recorder would refuse, because the
    recorder re-checks on the thread that owns the answer and raises
    `MeetingCaptureAbortedError` — a 409 — when it declines.
    """
    if not get_user_settings().meeting_consent_acknowledged:
        raise HTTPException(status_code=403, detail=_CONSENT_REQUIRED_DETAIL)
    if dictation_recorder is not None and dictation_recorder.is_recording:
        raise HTTPException(status_code=409, detail="A dictation recording is in progress")
    if recorder.is_busy:
        raise HTTPException(status_code=409, detail="Already recording")
    try:
        await recorder.start()
    except SystemAudioUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except MeetingCaptureAbortedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _meeting_status(recorder)


@router.post("/meeting/stop", response_model=MeetingStopResponse)
async def stop_meeting_recording(recorder: MeetingRecorder = Depends(get_meeting_recorder)):
    """End the recording and return the written file.

    The guard is `is_busy`, not `is_recording`, so a stop that arrives while
    the devices are still opening reaches `recorder.stop()` and is answered
    after that open rather than being refused before it. Like the other two
    guards it can only refuse: the recorder decides on its own thread and
    raises `MeetingCaptureAbortedError` when there is no file to return.

    Every 409 this endpoint can produce means nothing is being recorded and
    both devices are released — which is what lets the widget take its
    indicator down on one (`src/widget/meeting-toggle.ts`).

    Both `duration_seconds` and `truncated` are read before the await
    because the recorder returns to idle as part of stopping, and afterwards
    both read as the idle values.
    """
    if not recorder.is_busy:
        raise HTTPException(status_code=409, detail="Not recording")
    duration = recorder.duration_seconds
    truncated = recorder.truncated
    try:
        audio_path = await recorder.stop()
    except MeetingCaptureAbortedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return MeetingStopResponse(
        filename=audio_path.name,
        duration_seconds=duration,
        truncated=truncated,
    )


@router.get("/meeting/status", response_model=MeetingStatus)
async def meeting_recording_status(recorder: MeetingRecorder = Depends(get_meeting_recorder)):
    return _meeting_status(recorder)


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
