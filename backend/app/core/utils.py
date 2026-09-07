"""Server-Sent Event formatting for the audio and local-setup streaming endpoints."""

import json


def sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
