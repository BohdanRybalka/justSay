from abc import ABC, abstractmethod
from pathlib import Path


class AudioRecorder(ABC):
    """Contract: start recording → stop recording → get audio file."""

    @abstractmethod
    async def start(self) -> None:
        """Begin capturing audio."""

    @abstractmethod
    async def stop(self) -> Path:
        """Stop capturing and return path to the recorded WAV file."""

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Whether recording is currently active."""

    @property
    @abstractmethod
    def duration_seconds(self) -> float:
        """Elapsed recording time in seconds. 0 if not recording."""

    @property
    @abstractmethod
    def level_db(self) -> float:
        """Current audio input level in dBFS. -inf if silent or not recording."""
