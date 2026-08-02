"""The system-audio capture contract and its platform factory.

One abstract class with one implementation per platform, in the same shape as
the STT providers: nothing else in the audio package knows which OS it is
running on. See docs/adr/037-system-audio-capture-is-a-per-platform-source.md
and docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

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

    @property
    @abstractmethod
    def endpoint_name(self) -> str:
        """What is being captured, named the way the user would recognise it."""


def create_system_audio_source(
    settings: AudioSettings, platform_name: str | None = None
) -> SystemAudioSource | None:
    """The system-audio source for this platform.

    Returns None only when this platform has no capture path at all, which is
    a fact about the operating system and the same on every machine running
    it. A platform that *has* one and could not open it raises
    ``SystemAudioUnavailableError`` carrying the reason, because that is a
    fact about this machine and the user is the only person who can act on
    it. Collapsing the two into None is what made the caller tell a Windows
    user that meeting recording requires Windows.

    Both platform modules are imported here and nowhere else: `pyaudiowpatch`
    is a Windows-only wheel, and the macOS source reaches a helper binary that
    exists only inside a macOS bundle.

    `platform_name` is injectable so both branches are covered on the ubuntu
    CI runner.
    """
    platform_name = sys.platform if platform_name is None else platform_name

    try:
        if platform_name == "win32":
            from app.audio.windows_loopback import WindowsLoopbackSource

            return WindowsLoopbackSource(settings)

        if platform_name == "darwin":
            from app.audio.macos_tap import MacOSTapSource, resolve_audio_tap_path

            return MacOSTapSource(
                settings,
                resolve_audio_tap_path(
                    Path(sys.executable), settings.meeting_macos_tap_path
                ),
            )
    except SystemAudioUnavailableError:
        raise
    except Exception as failure:
        log.warning(
            "System-audio capture could not be opened on %s", platform_name, exc_info=True
        )
        raise SystemAudioUnavailableError(
            f"System audio capture could not be opened on this machine: {failure}"
        ) from failure

    return None
