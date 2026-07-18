from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class TranscriptionResult:
    text: str
    tokens_used: int | None = field(default=None)


class STTProvider(ABC):
    """Contract: Audio file in -> transcribed text out."""

    # Spec 028 Item 2 / ADR 018: locality is a property a provider declares
    # about itself, not a fact derived by probing the host platform.
    # `app.stt.is_local_provider()` reads this directly (a getattr, no I/O).
    # Overridden `True` on LocalSTTProvider, MLXWhisperSTTProvider, and
    # WhisperCppVulkanSTTProvider -- the three local, flat-sibling
    # implementations under this ABC. A provider that forgets the override
    # silently regresses to the pre-028 race (see docs/adr/018's Consequences
    # and this spec's "declared subclasses" test).
    is_local: ClassVar[bool] = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs) -> TranscriptionResult:
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
                - ``LocalSTTProvider`` / ``MLXWhisperSTTProvider``: translate
                  ``"auto"`` to ``language=None``, both providers' own native
                  auto-detect sentinel (faster-whisper / mlx-whisper).
                - ``WhisperCppVulkanSTTProvider``: forwards the literal string
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
            TranscriptionResult with text and optional token count.
        """

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Default: no-op.
        """
