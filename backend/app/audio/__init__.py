"""Audio module — microphone and system audio capture."""

from fastapi import Request

from app.audio.base import AudioRecorder
from app.audio.config import AudioSettings
from app.audio.meeting_recorder import MeetingRecorder
from app.audio.recorder import MicrophoneRecorder

__all__ = [
    "AudioRecorder",
    "AudioSettings",
    "MeetingRecorder",
    "MicrophoneRecorder",
    "get_active_meeting_recorder",
    "get_active_recorder",
    "get_meeting_recorder",
    "get_recorder",
]


def get_recorder(request: Request) -> MicrophoneRecorder:
    """FastAPI dependency — the app-lifetime MicrophoneRecorder.

    The instance is created once in main.py's lifespan startup and stored on
    app.state.recorder; this is the Depends() accessor routes use to reach
    it. See docs/adr/005-audio-recorder-di-lifespan.md.
    """
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None:
        raise RuntimeError(
            "app.state.recorder is not set — it is only created by main.py's "
            "lifespan startup. In tests, set "
            "app.dependency_overrides[get_recorder] instead of relying on "
            "the real recorder."
        )
    return recorder


def get_meeting_recorder(request: Request) -> MeetingRecorder:
    """FastAPI dependency — the app-lifetime MeetingRecorder.

    Same shape and same lifecycle as get_recorder: created once in main.py's
    lifespan startup, stored on app.state.meeting_recorder. See
    docs/adr/005-audio-recorder-di-lifespan.md.
    """
    recorder = getattr(request.app.state, "meeting_recorder", None)
    if recorder is None:
        raise RuntimeError(
            "app.state.meeting_recorder is not set — it is only created by "
            "main.py's lifespan startup. In tests, set "
            "app.dependency_overrides[get_meeting_recorder] instead of "
            "relying on the real recorder."
        )
    return recorder


def get_active_recorder(request: Request) -> MicrophoneRecorder | None:
    """The dictation recorder if one exists, otherwise None.

    Used only by the meeting endpoints to refuse starting while Instant
    Prompt holds the microphone. Unlike get_recorder it never raises: a
    missing recorder means nothing is dictating, which is exactly the answer
    the guard needs.
    """
    return getattr(request.app.state, "recorder", None)


def get_active_meeting_recorder(request: Request) -> MeetingRecorder | None:
    """The meeting recorder if one exists, otherwise None.

    The mirror image of get_active_recorder, used by the Instant Prompt
    endpoints to refuse starting while a meeting is being recorded.
    """
    return getattr(request.app.state, "meeting_recorder", None)
