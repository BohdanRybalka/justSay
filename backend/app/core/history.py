"""Transcript history — JSON-lines storage at ~/.justsay/history.jsonl."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, ValidationError

HISTORY_DIR = Path.home() / ".justsay"
HISTORY_PATH = HISTORY_DIR / "history.jsonl"
MAX_ENTRIES = 500

_lock = threading.Lock()


class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    language: str
    style: str
    raw_text: str
    cleaned_text: str
    duration_ms: int
    model_name: str | None = None
    tokens_used: int | None = None
    audio_duration_seconds: float | None = None
    word_count: int | None = None


def save_entry(
    raw_text: str,
    cleaned_text: str,
    duration_ms: int,
    language: str = "uk",
    style: str = "normal",
    model_name: str | None = None,
    tokens_used: int | None = None,
    audio_duration_seconds: float | None = None,
    word_count: int | None = None,
) -> HistoryEntry:
    """Append a new entry to history."""
    entry = HistoryEntry(
        id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        language=language,
        style=style,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        duration_ms=duration_ms,
        model_name=model_name,
        tokens_used=tokens_used,
        audio_duration_seconds=audio_duration_seconds,
        word_count=word_count,
    )

    with _lock:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        # Truncate if over limit
        _truncate_if_needed()

    return entry


def get_entries(limit: int = 50, offset: int = 0) -> list[HistoryEntry]:
    """Get history entries, newest first."""
    with _lock:
        entries = _read_all()
    entries.reverse()  # newest first
    return entries[offset : offset + limit]


def get_count() -> int:
    """Get total number of entries."""
    with _lock:
        return len(_read_all())


def delete_entry(entry_id: str) -> bool:
    """Delete a single entry by ID."""
    with _lock:
        entries = _read_all()
        filtered = [e for e in entries if e.id != entry_id]
        if len(filtered) == len(entries):
            return False
        _write_all(filtered)
        return True


def clear_all() -> int:
    """Delete all history. Returns number of deleted entries."""
    with _lock:
        count = len(_read_all())
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        return count


def _read_all() -> list[HistoryEntry]:
    """Read all entries from disk."""
    if not HISTORY_PATH.exists():
        return []
    entries = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(HistoryEntry.model_validate_json(line))
        except (json.JSONDecodeError, ValueError, ValidationError):
            continue  # skip corrupt lines
    return entries


def _write_all(entries: list[HistoryEntry]) -> None:
    """Rewrite the entire file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")


def _truncate_if_needed() -> None:
    """Keep only the last MAX_ENTRIES entries."""
    entries = _read_all()
    if len(entries) > MAX_ENTRIES:
        _write_all(entries[-MAX_ENTRIES:])
