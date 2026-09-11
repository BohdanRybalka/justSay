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
from app.audio.meeting_recorder import (
    MEETING_BUSY_DETAIL,
    MeetingCaptureAbortedError,
    MeetingCaptureEmptyError,
    MeetingRecorder,
    MeetingWriteFailedError,
)
from app.audio.recorder import MicrophoneRecorder, NotRecordingError
from app.audio.session import SessionMismatchError, SessionRef
from app.audio.system_source import SystemAudioUnavailableError
from app.core.utils import sse_event
from app.preferences.user_settings import get_user_settings

router = APIRouter()


class RecordingStatus(BaseModel):
    """`session_id` is the live capture's owner, or `null` when idle.

    Echoed rather than compared here: a caller that minted the id compares it
    locally, which is what lets a window tell its own abandoned start from
    somebody else's recording without the backend knowing anything about
    windows. It is an identity, not a secret — any caller holding the launch
    token can read it — so ADR 026's per-launch shared secret remains the
    security boundary.
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

    The Instant Prompt response model must stay untouched by this feature, and
    the meeting path reports one thing dictation does not: whether anything
    went wrong during the capture, named by a `CaptureIncident` token, or
    `null` when nothing did.
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
    """The dictation status, built in one place so `/start` and `/status`
    cannot answer with different fields."""
    return RecordingStatus(
        is_recording=recorder.is_recording,
        duration_seconds=recorder.duration_seconds,
        level_db=recorder.level_db,
        session_id=recorder.session_id,
    )


def _meeting_status(recorder: MeetingRecorder) -> MeetingStatus:
    """Build the response from one snapshot rather than five property reads.

    Read one at a time, the device thread can finish a stop between two of
    them and the response describes two different moments — a live meeting
    with no elapsed time and no endpoint. `syncMeetingIndicator` runs once at
    widget load and nothing polls after it, so such a response leaves a
    ticking indicator up for a call that has already ended.
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

    The body is optional and its absence is the pre-spec-119 contract: a
    request with no body at all, and one with `Content-Type: application/json`
    and an empty body, both bind `None` and start an unowned capture.
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

    Kept guarded although no frontend caller remains — the Settings
    microphone test moved to `/discard` and the widget dictates — because this
    is the documented Audio-In contract and an unguarded third mutating
    endpoint is exactly how the ownership race would come back.
    """
    if not recorder.is_recording:
        raise HTTPException(status_code=409, detail="Not recording")
    duration = recorder.duration_seconds
    try:
        audio_path = await recorder.stop(ref.session_id if ref else None)
    except SessionMismatchError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
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

    The session id is required here because there is no legacy caller to keep
    compatible, and because an unowned discard would be a way for any window
    to end any capture — the race this spec closes, reopened at a new
    endpoint.

    Both refusals are answers, which is what makes this the probe a client
    uses to find out whether a request it abandoned was ever processed: 200
    means the backend still held that session, so nothing downstream of the
    start had run; 403 and 409 mean it did not, so something else already
    happened to it.
    """
    try:
        dropped_seconds = await recorder.discard(ref.session_id)
    except SessionMismatchError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except NotRecordingError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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
        raise HTTPException(status_code=409, detail=MEETING_BUSY_DETAIL)
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

    Every 409, the one 410 and the one 507 this endpoint can produce mean
    nothing is being recorded and both devices are released — which is what
    lets the widget take its indicator down on all three
    (`src/widget/meeting-toggle.ts`). They are three codes rather than three
    wordings because a stop that found nothing recording, a meeting that
    captured nothing and a meeting whose audio was lost on the way to disk
    are different outcomes, and the widget must not describe any of them as
    a double click. 507 in particular replaces the 500 a failed write used to
    raise: the recorder is idle by then, but a 500 is indistinguishable from
    an unreachable backend, so the widget kept the indicator lit.

    `duration_seconds` and `capture_incident` arrive with the file, inside
    the `MeetingRecording` the write produces, rather than being read off the
    recorder: the harvest clears the live clock, so reading it here answers
    `0.0`, and a meeting started while this file is still being written owns
    the recorder's live incident by then.
    """
    if not recorder.is_busy:
        raise HTTPException(status_code=409, detail="Not recording")
    try:
        recording = await recorder.stop()
    except MeetingWriteFailedError as e:
        raise HTTPException(status_code=507, detail=str(e)) from e
    except MeetingCaptureEmptyError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e
    except MeetingCaptureAbortedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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
