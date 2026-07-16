"""Audio module — microphone and system audio capture."""

from fastapi import Request

from app.audio.base import AudioRecorder
from app.audio.config import AudioSettings
from app.audio.recorder import MicrophoneRecorder

__all__ = ["AudioRecorder", "AudioSettings", "MicrophoneRecorder", "get_recorder"]


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
