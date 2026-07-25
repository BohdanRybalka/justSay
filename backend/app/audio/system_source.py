"""The system-audio capture contract and its platform factory.

One abstract class with one implementation per platform, in the same shape as
the STT providers: nothing else in the audio package knows which OS it is
running on. See docs/adr/037-system-audio-capture-is-a-per-platform-source.md.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from app.audio.config import AudioSettings

log = logging.getLogger(__name__)

BlockSink = Callable[[float, np.ndarray], None]


class SystemAudioUnavailableError(RuntimeError):
    """No system-audio capture exists on this platform, or no device was found."""


class SystemAudioSource(ABC):
    """Contract: start delivering timestamped mono blocks → stop."""

    @abstractmethod
    def start(self, on_block: BlockSink) -> None:
        """Begin capturing, calling `on_block(arrival_monotonic, mono_block)`."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device. Safe to call when not started."""

    @property
    @abstractmethod
    def native_sample_rate(self) -> int:
        """The rate the device actually delivers at, before any resampling."""


def create_system_audio_source(settings: AudioSettings) -> SystemAudioSource | None:
    """The system-audio source for this platform, or None if there is none.

    Returns None rather than raising, so the caller decides what an absent
    source means. `app.audio.windows_loopback` is imported here and nowhere
    else, which keeps `pyaudiowpatch` — a Windows-only wheel — off the import
    path on macOS and on the ubuntu CI runner entirely.
    """
    if sys.platform != "win32":
        return None

    try:
        from app.audio.windows_loopback import WindowsLoopbackSource

        return WindowsLoopbackSource(settings)
    except Exception:
        log.warning(
            "No WASAPI loopback source available — meeting recording is off on this machine",
            exc_info=True,
        )
        return None
