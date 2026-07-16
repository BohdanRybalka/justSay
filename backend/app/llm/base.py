from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract: Raw text in -> processed text out."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the current model."""

    @abstractmethod
    async def process(self, text: str, system_prompt: str, task: str = "dictation_cleanup") -> str:
        """Process text with a system prompt.

        Args:
            text: Raw transcribed text.
            system_prompt: Instructions for text processing.
            task: Task name resolved to a generation profile
                (temperature/top_p/max_tokens) via `app.llm.tasks.get_task_profile`.

        Returns:
            Processed text.
        """

    def cleanup(self) -> None:
        """Release resources (model memory, connections).

        Called on mode switch and app shutdown. Default: no-op.
        """
