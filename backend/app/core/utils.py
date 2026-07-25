"""Shared utilities used across routers and setup modules."""

import json

from fastapi import HTTPException, UploadFile


def sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def read_upload_with_limit(file: UploadFile, max_size: int) -> bytes:
    """Stream-read an UploadFile in 64 KB chunks, raising HTTP 413 when exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {max_size // 1024 // 1024}MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def bytes_to_mb(b: int) -> int:
    return b // (1024 * 1024)


def bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)
