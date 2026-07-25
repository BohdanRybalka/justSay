from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class TranscriptionResult:
    text: str
    tokens_used: int | None = field(default=None)
    detected_language: str | None = field(default=None)
    no_speech_prob: float | None = field(default=None)


def normalize_detected_language(raw: str | None) -> str | None:
    """Normalize a provider-reported language into a lowercase ISO-639-1
    code, or ``None`` when unrecognised or empty.

    Handles the raw shapes providers actually return:
      - An already-two-letter code, any case (``"EN"``) -> lowercased as-is.
      - A region-tagged code (``"en-US"``, ``"pt_BR"``) -> the primary subtag.
      - A full English language name (``"english"``, ``"Ukrainian"``) ->
        looked up against `app.pipeline.prompts.LANGUAGE_NAMES`, reversed —
        covers at minimum the codes in that table.

    Never passes an unrecognised value through — a garbage code reaching
    `entries.language` / the Words tab's ``by_language`` bucket is worse
    than the ``"auto"`` sentinel it replaces. The providers whose SDKs may
    report a full language name rather than a code (Groq's ``verbose_json``
    response, closed-source and undocumented) fall back to `None` here,
    which downstream (`process_audio`) means "keep the auto sentinel" —
    not an error.
    """
    if not raw or not raw.strip():
        return None
    candidate = raw.strip().lower().replace("_", "-")
    primary = candidate.split("-")[0]

    if len(primary) == 2 and primary.isalpha():
        return primary

    from app.pipeline.prompts import LANGUAGE_NAMES

    name_to_code = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}
    return name_to_code.get(candidate) or name_to_code.get(primary)


def coerce_no_speech_prob(value) -> float | None:
    """Coerce one raw ``no_speech_prob`` value to a float, or ``None``.

    ``bool`` is excluded explicitly: it is a subclass of ``int``, so a
    stubbed ``"no_speech_prob": false`` would otherwise read as 0.0 — a
    confident "definitely speech" verdict invented out of a missing value,
    which the pipeline's post-model gate would then trust.

    Shared by `min_no_speech_prob` (the ``verbose_json`` readers) and
    `LocalSTTProvider._transcribe`'s lazy-generator loop, which cannot reuse
    the aggregate helper but must not drift from its defensiveness.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def min_no_speech_prob(segments) -> float | None:
    """Minimum ``no_speech_prob`` across ``segments``, or ``None``.

    Shared by the two providers that read this off a ``verbose_json``
    payload (`WhisperCppServerSTTProvider`, `GroqWhisperSTTProvider`) — one
    defensive reader rather than two drifting copies.

    Deliberately total: ``None``/non-sequence input, an empty list, segments
    that are neither dicts nor attribute-objects, and a missing or
    non-numeric ``no_speech_prob`` field all yield ``None`` rather than
    raising. whisper.cpp builds vary in whether they populate the field at
    all, and Groq's SDK returns attribute-objects or dicts depending on
    version — a shape surprise must fail OPEN (keep the transcription), never
    break a transcription that already succeeded.

    Only ``list``/``tuple`` are accepted by design: any other sequence type a
    future provider version might return (generator, pydantic sequence) fails
    open to ``None`` rather than being consumed speculatively.

    Per-value coercion — including the ``bool`` exclusion — is delegated to
    `coerce_no_speech_prob`.
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
    """Contract: Audio file in -> transcribed text out."""

    is_local: ClassVar[bool] = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV, 16kHz, mono).
            language: BCP-47 language code, or the sentinel ``"auto"`` to
                request the provider's own native auto-detect mechanism
                instead of assuming a language. Each concrete provider
                translates ``"auto"`` differently:
                - ``GroqWhisperSTTProvider``: omits the ``language`` kwarg
                  entirely from the Groq SDK call (mirrors the SDK's own
                  ``Omit`` default).
                - ``GeminiSTTProvider``: swaps the prompt's language clause
                  for an instruction to detect the spoken language itself.
                - ``LocalSTTProvider``: translates ``"auto"`` to
                  ``language=None``, faster-whisper's own native auto-detect
                  sentinel.
                - ``WhisperCppServerSTTProvider``: forwards the literal string
                  ``"auto"`` unchanged — whisper.cpp's core library treats it
                  as its own native auto-detect sentinel, so no translation
                  is needed.
            **kwargs: Provider-specific extensions. Currently recognised:
                - ``style`` ("normal" | "ai_prompt"): Gemini uses it to select
                  between a faithful transcription prompt and a structuring prompt.
                  Groq / local providers ignore it.
                - ``audio_duration`` (float, seconds): when known, the local
                  provider uses it to pick a latency-vs-accuracy beam_size
                  (1 for short clips, 5 for long). Cloud providers ignore it.

        Returns:
            TranscriptionResult with text, optional token count, and
            ``detected_language`` (normalized ISO-639-1 code or ``None``).
            Providers populate ``detected_language`` unevenly:
                - ``LocalSTTProvider``: always, from the underlying whisper
                  model's own language field (`TranscriptionInfo.language`)
                  — populated whether or not ``language`` was ``"auto"``.
                - ``WhisperCppServerSTTProvider`` / ``GroqWhisperSTTProvider``:
                  only when ``language == "auto"`` — both escalate to a
                  richer wire format (``verbose_json``) on that path only,
                  keeping their current format/parsing unchanged for
                  explicit-language requests. Always ``None`` otherwise.
                - ``GeminiSTTProvider``: always ``None`` — no structured
                  language field exists at any setting.

            ``no_speech_prob`` (min across segments) is populated just as
            unevenly, and for the same wire-format reasons:
                - ``LocalSTTProvider``: always — faster-whisper's
                  ``Segment.no_speech_prob`` is on every segment.
                - ``WhisperCppServerSTTProvider`` / ``GroqWhisperSTTProvider``:
                  only when ``language == "auto"`` (the only path that uses
                  ``verbose_json``), and read defensively there — whisper.cpp
                  builds vary in whether the field carries a live value, and a
                  missing/stubbed field yields ``None``, never an exception.
                - ``GeminiSTTProvider``: always ``None`` — no structured
                  no-speech signal at any setting (ADR 016).
        """

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Default: no-op.
        """
