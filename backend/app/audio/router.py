"""The HTTP surface of dictation and meeting recording.

The `HTTPException` raises left here are this layer's own guards -- consent,
"already recording", "not recording" -- decided from state the router can read
before it calls anything. Every refusal the recorders themselves produce
carries its own status on its exception class and reaches the client through
the error handler, so those statuses are documented where they are raised.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.audio.dependencies import (
    get_active_meeting_recorder,
    get_active_recorder,
    get_meeting_recorder,
    get_recorder,
)
from app.audio.meeting_recorder import MEETING_BUSY_DETAIL, MeetingRecorder
from app.audio.recorder import MicrophoneRecorder
from app.audio.session import SessionRef
from app.core.utils import sse_event
from app.preferences.user_settings import get_user_settings

router = APIRouter()


class RecordingStatus(BaseModel):
    """`session_id` is the live capture's owner, or `null` when idle.

    Echoed rather than compared here: a caller that minted the id compares it
    locally. It is an identity, not a secret — ADR 026's per-launch shared
    secret remains the security boundary.
    """

    is_recording: bool
    duration_seconds: float
    level_db: float
    session_id: str | None = None


class StopResponse(BaseModel):
    filename: str
    duration_seconds: float


class DiscardResponse(BaseModel):
    """What a discarded capture leaves behind: its length, and no file.

    Deliberately not `StopResponse` without the filename — there is no file,
    and a model with an empty `filename` would invite a caller to look for one.
    """

    duration_seconds: float


class MeetingStopResponse(BaseModel):
    """Deliberately separate from StopResponse.

    The meeting path reports what dictation does not: whether anything went
    wrong during the capture, named by a `CaptureIncident` token, or `null`.
    """

    filename: str
    duration_seconds: float
    capture_incident: str | None


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
    capture_incident: str | None


_CONSENT_REQUIRED_DETAIL = (
    "Meeting recording has not been acknowledged — open Settings → General and "
    "confirm you are responsible for obtaining the participants' consent"
)


def _recording_status(recorder: MicrophoneRecorder) -> RecordingStatus:
    """The dictation status, built once so `/start` and `/status` cannot
    answer with different fields.
    """
    return RecordingStatus(
        is_recording=recorder.is_recording,
        duration_seconds=recorder.duration_seconds,
        level_db=recorder.level_db,
        session_id=recorder.session_id,
    )


def _meeting_status(recorder: MeetingRecorder) -> MeetingStatus:
    """Build the response from one snapshot rather than five property reads.

    Read one at a time, the device thread can finish a stop between two of
    them and the response would describe two different moments — a live
    meeting with no elapsed time and no endpoint.
    """
    snapshot = recorder.status_snapshot()
    return MeetingStatus(
        is_recording=snapshot.is_recording,
        duration_seconds=snapshot.duration_seconds,
        level_db=snapshot.level_db,
        system_endpoint=snapshot.system_endpoint,
        system_level_db=snapshot.system_level_db,
        capture_incident=(
            None if snapshot.capture_incident is None else snapshot.capture_incident.value
        ),
    )


@router.post("/start", response_model=RecordingStatus)
async def start_recording(
    ref: SessionRef | None = None,
    recorder: MicrophoneRecorder = Depends(get_recorder),
    meeting_recorder: MeetingRecorder | None = Depends(get_active_meeting_recorder),
):
    """Open the microphone, recording the caller's session id as its owner.

    The body is optional: no body at all, and `Content-Type: application/json`
    with an empty body, both bind `None` and start an unowned capture.
    """
    if meeting_recorder is not None and meeting_recorder.is_busy:
        raise HTTPException(status_code=409, detail="A meeting recording is in progress")
    if recorder.is_recording:
        raise HTTPException(status_code=409, detail="Already recording")
    await recorder.start(ref.session_id if ref else None)
    return _recording_status(recorder)


@router.post("/stop", response_model=StopResponse)
async def stop_recording(
    ref: SessionRef | None = None,
    recorder: MicrophoneRecorder = Depends(get_recorder),
):
    """Harvest the capture to a WAV, refusing a stranger's session.

    Kept guarded although no frontend caller remains: this is the documented
    Audio-In contract, and an unguarded mutating endpoint is how the
    ownership race would come back.
    """
    if not recorder.is_recording:
        raise HTTPException(status_code=409, detail="Not recording")
    duration = recorder.duration_seconds
    audio_path = await recorder.stop(ref.session_id if ref else None)
    return StopResponse(
        filename=audio_path.name,
        duration_seconds=duration,
    )


@router.post("/discard", response_model=DiscardResponse)
async def discard_recording(
    ref: SessionRef,
    recorder: MicrophoneRecorder = Depends(get_recorder),
):
    """End the caller's own capture and write nothing.

    The session id is required: an unowned discard would let any window end
    any capture. Both refusals are answers, so a client can use this to learn
    whether a start it abandoned was processed — 200 means it was not.
    """
    dropped_seconds = await recorder.discard(ref.session_id)
    return DiscardResponse(duration_seconds=dropped_seconds)


@router.get("/status", response_model=RecordingStatus)
async def recording_status(recorder: MicrophoneRecorder = Depends(get_recorder)):
    return _recording_status(recorder)


@router.post("/meeting/start", response_model=MeetingStatus)
async def start_meeting_recording(
    recorder: MeetingRecorder = Depends(get_meeting_recorder),
    dictation_recorder: MicrophoneRecorder | None = Depends(get_active_recorder),
):
    """Begin capturing the microphone and the system output together.

    Answers 403 until the meeting disclosure has been acknowledged (ADR 040),
    and 501 on a platform with no system-audio path. The `is_busy` guard can
    only refuse — the recorder re-checks on the thread that owns the answer.
    """
    if not get_user_settings().meeting_consent_acknowledged:
        raise HTTPException(status_code=403, detail=_CONSENT_REQUIRED_DETAIL)
    if dictation_recorder is not None and dictation_recorder.is_recording:
        raise HTTPException(status_code=409, detail="A dictation recording is in progress")
    if recorder.is_busy:
        raise HTTPException(status_code=409, detail=MEETING_BUSY_DETAIL)
    await recorder.start()
    return _meeting_status(recorder)


@router.post("/meeting/stop", response_model=MeetingStopResponse)
async def stop_meeting_recording(recorder: MeetingRecorder = Depends(get_meeting_recorder)):
    """End the recording and return the written file.

    Guarded on `is_busy`, not `is_recording`, so a stop arriving while the
    devices are still opening is answered after that open. Every 409, the 410
    and the 507 all mean nothing is recording and both devices are released.
    """
    if not recorder.is_busy:
        raise HTTPException(status_code=409, detail="Not recording")
    recording = await recorder.stop()
    return MeetingStopResponse(
        filename=recording.path.name,
        duration_seconds=recording.duration_seconds,
        capture_incident=(
            None if recording.incident is None else recording.incident.value
        ),
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
