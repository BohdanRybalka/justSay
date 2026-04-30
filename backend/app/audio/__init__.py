"""Audio module — microphone and system audio capture."""

from app.audio.base import AudioRecorder
from app.audio.config import AudioSettings
from app.audio.recorder import MicrophoneRecorder

__all__ = ["AudioRecorder", "AudioSettings", "MicrophoneRecorder", "get_recorder"]

_recorder_instance: MicrophoneRecorder | None = None


def get_recorder() -> MicrophoneRecorder:
    """Shared singleton accessor for the microphone recorder."""
    global _recorder_instance
    if _recorder_instance is None:
        from app.core.config import settings

        _recorder_instance = MicrophoneRecorder(settings.audio)
    return _recorder_instance
