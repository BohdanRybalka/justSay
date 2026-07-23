"""LLM module — retains only ``LLMSettings``.

``LLMSettings.mode`` is one half of the ``(stt.mode, llm.mode)`` key that
gates embedding-provider eligibility (see ``app.embeddings``). The dead LLM
text-processing stack (providers, factory, ``/llm/*`` router) was removed in
Spec 045 / ADR 029.
"""

from app.llm.config import LLMSettings

__all__ = ["LLMSettings"]
