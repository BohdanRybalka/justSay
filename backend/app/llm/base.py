from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract: Raw text in -> processed text out."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def process(self, text: str, system_prompt: str, temperature: float = 0.1) -> str:
        """Process text with a system prompt.

        Args:
            text: Raw transcribed text.
            system_prompt: Instructions for text processing.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Processed text.
        """

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Default: no-op.
        """
