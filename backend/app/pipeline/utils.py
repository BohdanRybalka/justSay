"""Shared pipeline utilities — audio duration detection, etc."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def detect_duration(audio_path: Path) -> float | None:
    """Return duration in seconds, or None if the file can't be inspected.

    ``None`` is a valid result — callers route unknown-length audio to Gemini
    (the safe default that handles everything).
    """
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.duration)
    except Exception as e:  # corrupt file, unsupported container, soundfile missing
        log.warning("Duration detection failed for %s: %s", audio_path, e)
        return None
