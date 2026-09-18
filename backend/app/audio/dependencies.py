"""FastAPI wiring for the audio package, and nothing else.

Dependency injection is the only reason this module imports ``fastapi``, and
the ``Depends()`` accessors below are all of it. Capture logic, DSP and
configuration live in their own modules and stay web-framework free, so
importing one of them does not drag the HTTP stack into the process.
Anything that is not dependency injection belongs elsewhere.
"""

from typing import TypeVar

from fastapi import Request

from app.audio.meeting_recorder import MeetingRecorder
from app.audio.recorder import MicrophoneRecorder

Recorder = TypeVar("Recorder")


def _require(recorder: Recorder | None, attribute: str, dependency: str) -> Recorder:
    """The recorder, or a ``RuntimeError`` naming why there is none.

    Raised when lifespan startup did not run, or when a guarded startup step could not build the
    recorder and let the backend carry on without it; that step's WARNING says what failed.
    """
    if recorder is None:
        raise RuntimeError(
            f"app.state.{attribute} is not set — main.py's lifespan startup "
            f"either did not run or could not build it, and the startup log "
            f"says which. In tests, set "
            f"app.dependency_overrides[{dependency}] instead of relying on "
            f"the real recorder."
        )
    return recorder


def get_recorder(request: Request) -> MicrophoneRecorder:
    """FastAPI dependency — the app-lifetime ``MicrophoneRecorder`` (ADR 005).

    Created once in lifespan startup and stored on ``app.state.recorder``; raises when absent.
    """
    return _require(get_active_recorder(request), "recorder", "get_recorder")


def get_meeting_recorder(request: Request) -> MeetingRecorder:
    """FastAPI dependency — the app-lifetime ``MeetingRecorder``, when there is one (ADR 005).

    Building it is a guarded startup step, because dictation, History and Settings all work
    without it, so an install where it failed serves every other endpoint and answers 500 here.
    """
    return _require(
        get_active_meeting_recorder(request), "meeting_recorder", "get_meeting_recorder"
    )


def get_active_recorder(request: Request) -> MicrophoneRecorder | None:
    """The dictation recorder if one exists, otherwise ``None``.

    Never raises: a missing recorder means nothing is dictating, which is the answer the meeting
    endpoints need to refuse starting while Instant Prompt holds the microphone.
    """
    return getattr(request.app.state, "recorder", None)


def get_active_meeting_recorder(request: Request) -> MeetingRecorder | None:
    """The meeting recorder if one exists, otherwise None.

    The mirror image of get_active_recorder, used by the Instant Prompt
    endpoints to refuse starting while a meeting is being recorded.
    """
    return getattr(request.app.state, "meeting_recorder", None)
