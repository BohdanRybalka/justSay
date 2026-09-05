"""Server-Sent Event formatting, plus two byte-size conversions.

``sse_event`` is used by the audio and local-setup streaming endpoints;
``bytes_to_mb`` and ``bytes_to_gb`` format the resource-report numbers. The
two concerns are unrelated and have disjoint callers.
"""

import json


def sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def bytes_to_mb(b: int) -> int:
    return b // (1024 * 1024)


def bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)
