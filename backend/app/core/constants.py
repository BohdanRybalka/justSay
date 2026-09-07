"""Shared backend constants.

Single source of truth for limits and identifiers used in more than one module.
Implementation-detail constants (e.g. cache TTLs) intentionally stay local to
their owning module.
"""

MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024

GROQ_TIMEOUT_SECONDS: float = 10.0

GEMINI_TIMEOUT_SECONDS: float = 300.0

GEMINI_EMBEDDING_TIMEOUT_SECONDS: float = 30.0

MASKED_API_KEY: str = "***"
