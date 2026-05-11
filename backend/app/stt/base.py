from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptionResult:
    text: str
    tokens_used: int | None = field(default=None)


class STTProvider(ABC):
    """Contract: Audio file in -> transcribed text out."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV, 16kHz, mono).
            language: BCP-47 language code.
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
