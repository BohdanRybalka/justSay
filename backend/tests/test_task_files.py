"""The task tracker under docs/tasks/ must stay parseable and unambiguous.

`docs/` is gitignored, so no CI job can ever see these files and nothing fails a
build when one drifts -- the same reason `test_adr_numbering.py` exists and the
same remedy: a local test that runs on every `pytest` invocation.

What it enforces is exactly what an audit on 2026-07-25 found missing. Entries
had no ids, so nothing could be referred to; no creation dates, which left
`CLAUDE.md`'s "a debt round opens when the oldest is more than 14 days old" rule
unenforceable; and blocks stated in prose, one of which turned out to rest on
code that did not exist. A block that must name its obstacle and a deferral that
must name its trigger are both falsifiable by a reader.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

TASKS_DIR = Path(__file__).resolve().parents[2] / "docs" / "tasks"
SPECS_DIR = Path(__file__).resolve().parents[2] / "specs"

LIVE_FILES = ("bugs.md", "backlog.md", "tech-debt.md")
ARCHIVE_FILE = "done.md"

STATUSES = {"todo", "in-progress", "blocked", "deferred", "wontfix", "done"}

_HEADING_RE = re.compile(r"^### JS-(\d+) — (.+)$", re.M)
_FIELD_RE = re.compile(r"\*\*(Status|Created|Updated|Trigger|Blocked by):\*\*\s*([^·\n]+)")
_ARCHIVE_ID_RE = re.compile(r"^- \[x\] \*\*JS-(\d+)\*\*", re.M)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

pytestmark = pytest.mark.skipif(
    not TASKS_DIR.is_dir(),
    reason="docs/tasks/ is gitignored and absent from clean checkouts",
)


def _entries(filename: str) -> list[dict]:
    """Every `### JS-N — Title` block in one file, with its metadata fields."""
    text = (TASKS_DIR / filename).read_text(encoding="utf-8")
    headings = list(_HEADING_RE.finditer(text))
    parsed = []
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[match.end() : end]
        fields = {key: value.strip() for key, value in _FIELD_RE.findall(body)}
        parsed.append(
            {
                "id": int(match.group(1)),
                "title": match.group(2).strip(),
                "file": filename,
                "fields": fields,
            }
        )
    return parsed


def _live_entries() -> list[dict]:
    return [entry for name in LIVE_FILES for entry in _entries(name)]


def test_every_live_entry_has_a_unique_id():
    seen: dict[int, str] = {}
    duplicates = []
    for entry in _live_entries():
        origin = f"{entry['file']}:JS-{entry['id']}"
        if entry["id"] in seen:
            duplicates.append(f"JS-{entry['id']} in both {seen[entry['id']]} and {origin}")
        seen[entry["id"]] = origin

    assert not duplicates, (
        "Task ids must be unique across every file -- an id is allocated once and never "
        f"reused (docs/tasks/README.md → Ids): {duplicates}"
    )


def test_no_live_id_collides_with_the_archive():
    live = {entry["id"] for entry in _live_entries()}
    archived = {
        int(m.group(1))
        for m in _ARCHIVE_ID_RE.finditer((TASKS_DIR / ARCHIVE_FILE).read_text(encoding="utf-8"))
    }

    collisions = sorted(live & archived)

    assert not collisions, (
        f"These ids appear both as live work and as closed work: {collisions}. An id is never "
        "reassigned, so a collision means one of the two is mislabelled."
    )


def test_every_live_entry_declares_a_known_status():
    offenders = [
        (entry["file"], entry["id"], entry["fields"].get("Status"))
        for entry in _live_entries()
        if entry["fields"].get("Status") not in STATUSES
    ]

    assert not offenders, (
        f"Unknown or missing Status: {offenders}. Allowed: {sorted(STATUSES)} "
        "(docs/tasks/README.md → Statuses)."
    )


def test_blocked_entries_name_their_obstacle():
    """A block nobody can check is indistinguishable from an invented one.

    The audit that produced this format found an entry whose entire stated
    rationale rested on a code path that did not exist, and several described as
    blocked that were really deliberate deferrals.
    """
    offenders = [
        (entry["file"], f"JS-{entry['id']}")
        for entry in _live_entries()
        if entry["fields"].get("Status") == "blocked" and not entry["fields"].get("Blocked by")
    ]

    assert not offenders, (
        f"`blocked` requires a non-empty **Blocked by:** naming the obstacle: {offenders}"
    )


def test_deferred_entries_name_their_trigger():
    offenders = [
        (entry["file"], f"JS-{entry['id']}")
        for entry in _live_entries()
        if entry["fields"].get("Status") == "deferred" and not entry["fields"].get("Trigger")
    ]

    assert not offenders, (
        f"`deferred` requires a non-empty **Trigger:** naming what would reopen it: {offenders}"
    )


def test_every_live_entry_carries_parseable_dates():
    offenders = []
    for entry in _live_entries():
        for field in ("Created", "Updated"):
            value = entry["fields"].get(field, "")
            if not _DATE_RE.match(value):
                offenders.append((entry["file"], f"JS-{entry['id']}", field, value))
                continue
            try:
                date.fromisoformat(value)
            except ValueError:
                offenders.append((entry["file"], f"JS-{entry['id']}", field, value))

    assert not offenders, f"Created/Updated must be a real YYYY-MM-DD date: {offenders}"


def test_updated_never_precedes_created():
    offenders = [
        (entry["file"], f"JS-{entry['id']}")
        for entry in _live_entries()
        if date.fromisoformat(entry["fields"]["Updated"])
        < date.fromisoformat(entry["fields"]["Created"])
    ]

    assert not offenders, f"Updated is earlier than Created: {offenders}"


def test_no_live_id_is_at_or_above_the_next_free_number():
    """Ids and spec directories draw from one counter, so a live entry can never
    hold a number a spec has already taken."""
    spec_numbers = {
        int(p.name[:3]) for p in SPECS_DIR.iterdir() if p.is_dir() and p.name[:3].isdigit()
    }
    clashes = sorted(entry["id"] for entry in _live_entries() if entry["id"] in spec_numbers)

    assert not clashes, (
        f"These ids are held by both a live entry and a specs/ directory: {clashes}. "
        "One counter, allocated once (docs/tasks/README.md → Ids)."
    )
