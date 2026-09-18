from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from app.stt.languages import LANGUAGE_NAMES


@dataclass
class TranscriptionResult:
    text: str
    tokens_used: int | None = field(default=None)
    detected_language: str | None = field(default=None)
    no_speech_prob: float | None = field(default=None)


LOAD_FAILED_WITHOUT_A_MESSAGE = "The local engine failed to load and gave no reason."


def latched_load_error(exc: BaseException) -> str:
    """The sentence `GET /stt/local/status`'s `last_error` shows for a failed load.

    Never empty and never a class name; an exception whose ``str()`` is empty
    yields a fixed fallback sentence instead.
    """
    return str(exc) or LOAD_FAILED_WITHOUT_A_MESSAGE


def normalize_detected_language(raw: str | None) -> str | None:
    """Normalize a provider-reported language into a lowercase ISO-639-1 code.

    Accepts a two-letter code in any case, a region-tagged code (``en-US``) or a
    full English name from `LANGUAGE_NAMES`; anything else returns ``None``.
    """
    if not raw or not raw.strip():
        return None
    candidate = raw.strip().lower().replace("_", "-")
    primary = candidate.split("-")[0]

    if len(primary) == 2 and primary.isalpha():
        return primary

    name_to_code = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}
    return name_to_code.get(candidate) or name_to_code.get(primary)


def clean_transcript_text(raw: str | None) -> str:
    """Coerce one provider's raw transcript into a stripped ``str``.

    ``None`` becomes ``""``. It coerces and nothing else: a provider discards a
    transcript through `TranscriptionResult` or by raising, never through text.
    """
    return raw.strip() if raw else ""


def coerce_no_speech_prob(value) -> float | None:
    """Coerce one raw ``no_speech_prob`` value to a float, or ``None``.

    ``True``/``False`` yield ``None`` rather than 1.0/0.0, so a stubbed
    boolean field cannot read as a confident verdict.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def min_no_speech_prob(segments) -> float | None:
    """Minimum ``no_speech_prob`` across ``segments``, or ``None``.

    Total: only ``list``/``tuple`` are read, and an empty list, an unknown
    segment shape or a missing field yield ``None`` rather than raising.
    """
    if not isinstance(segments, (list, tuple)):
        return None

    values: list[float] = []
    for segment in segments:
        if isinstance(segment, dict):
            raw = segment.get("no_speech_prob")
        else:
            raw = getattr(segment, "no_speech_prob", None)
        probability = coerce_no_speech_prob(raw)
        if probability is not None:
            values.append(probability)
    return min(values) if values else None


class STTProvider(ABC):
    """Contract: Audio file in -> transcribed text out.

    A local provider owes the further members that the factory enforces through
    :data:`app.stt.local_factory.LOCAL_STATUS_CONTRACT` (ADR 075).
    """

    is_local: ClassVar[bool] = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe ``audio_path`` (WAV, 16 kHz, mono) into text.

        ``language`` is a BCP-47 code or ``"auto"``; ``**kwargs`` carries an
        optional ``audio_duration`` in seconds. Unreported fields are ``None``.
        """

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Default: no-op.
        """
