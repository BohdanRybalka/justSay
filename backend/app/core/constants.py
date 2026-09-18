"""Shared backend constants.

Single source of truth for limits and identifiers shared across the backend.
Implementation-detail constants such as cache TTLs stay with the module that
owns them.
"""

MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024

GROQ_TIMEOUT_SECONDS: float = 10.0

GEMINI_TIMEOUT_SECONDS: float = 300.0

GEMINI_EMBEDDING_TIMEOUT_SECONDS: float = 30.0

MASKED_API_KEY: str = "***"
