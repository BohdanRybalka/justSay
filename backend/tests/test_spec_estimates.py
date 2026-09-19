"""Every spec at or above SCANNED_FROM derives its size estimate rather than asserting one.

Fails a missing `## Size estimate` section, a sourceless or non-integer row, a missing
`Review rounds` row, rows that do not sum, a landed file carrying no `Landed:` line in
that section, and CLAUDE.md or a document restating its re-score multiplier stating a
value other than RESCORE_MULTIPLIER (ADR 080).

specs/, .claude/ and CLAUDE.md are gitignored, so this gate runs locally only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = REPO_ROOT / "specs"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
MULTIPLIER_RESTATEMENTS = (
    REPO_ROOT / ".claude" / "commands" / "fix.md",
    REPO_ROOT / "specs" / "_templates" / "plan.md",
)

SCANNED_FROM = 188
RESCORE_MULTIPLIER = 2.0
ARBITRARY_SUM_TOLERANCE = 0.10
MIN_SOURCE_CHARS = 15
MIN_TABLE_ROWS = 2

SECTION_HEADING = "## Size estimate"
TABLE_HEADER = ("Term", "Lines", "Source")
REVIEW_ROUND_TERM = "Review rounds"
LANDED_STATUSES = frozenset({"pr-open", "done"})
SIZE_GATE_HEADING = "**The size gate — measured, not felt.**"

_SPEC_DIR_RE = re.compile(r"^(\d{3})-")
_SECTION_RE = re.compile(rf"^{re.escape(SECTION_HEADING)}\s*$", re.M)
_NEAR_SECTION_RE = re.compile(r"^#{1,6}[ 	]*size estimate.*$", re.M | re.I)
_STATUS_RE = re.compile(r"^>?\s*\*\*Status:\*\*\s*([a-z-]+)", re.M)
_LANDED_RE = re.compile(r"^Landed:.*?\d+.*?git diff --shortstat", re.M)
_INT_RE = re.compile(r"\d+")
_RESCORE_RE = re.compile(r"If the real diff lands more than (\d+(?:\.\d+)?)×")
_RESTATED_RESCORE_RE = re.compile(r"(?:more than|over|past)\s+(\d+(?:\.\d+)?)×", re.I)

pytestmark = pytest.mark.skipif(
    not SPECS_DIR.is_dir(),
    reason="specs/ is gitignored and absent from clean checkouts",
)


def _scanned_files() -> list[Path]:
    """Every plan.md and fix.md under a spec directory numbered SCANNED_FROM or above.

    Directories numbered below SCANNED_FROM are out of scope (ADR 080).
    """
    found: list[Path] = []
    for directory in sorted(SPECS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        match = _SPEC_DIR_RE.match(directory.name)
        if match is None or int(match.group(1)) < SCANNED_FROM:
            continue
        for name in ("plan.md", "fix.md"):
            path = directory / name
            if path.is_file():
                found.append(path)
    return found


def _size_estimate_block(text: str) -> str | None:
    """The `## Size estimate` section, or None when the file carries no such heading.

    The heading matches as a whole line only, never as a substring of surrounding prose.
    """
    match = _SECTION_RE.search(text)
    if match is None:
        return None
    rest = text[match.end() :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _split_cells(row: str) -> list[str]:
    """The row's cells, reading a backslash-escaped pipe as literal text."""
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in row:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
    cells.append("".join(current))
    return cells


def _table_rows(block: str) -> list[tuple[str, ...]]:
    """Every pipe-delimited row of the block, with the `| --- |` separators dropped."""
    rows: list[tuple[str, ...]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1):
            continue
        cells = tuple(cell.strip() for cell in _split_cells(stripped[1:-1]))
        if all(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _is_total(term: str) -> bool:
    return term.strip("* ").lower() == "total"


def _body_rows(block: str) -> list[tuple[str, ...]]:
    """The table's rows below the header row, with the Total row removed."""
    return [row for row in _table_rows(block)[1:] if row and not _is_total(row[0])]


def _total_row(block: str) -> tuple[str, ...] | None:
    for row in _table_rows(block):
        if row and _is_total(row[0]):
            return row
    return None


def _first_int(cell: str) -> int | None:
    match = _INT_RE.search(cell)
    return int(match.group()) if match else None


def _missing_section_offenders(label: str, text: str) -> list[str]:
    if _size_estimate_block(text) is not None:
        return []
    near = _NEAR_SECTION_RE.search(text)
    reason = "An estimate is a table, not a number (CLAUDE.md -> Workflow -> Routing)."
    if near is None:
        return [f"{label}: no `{SECTION_HEADING}` heading. {reason}"]
    return [
        f"{label}: found `{near.group().strip()}`, but the heading must read exactly "
        f"`{SECTION_HEADING}` -- two hashes, nothing after it. {reason}"
    ]


def _table_shape_offenders(label: str, text: str) -> list[str]:
    block = _size_estimate_block(text)
    if block is None:
        return []
    rows = _table_rows(block)
    offenders = []
    header = " | ".join(TABLE_HEADER)
    if not rows or rows[0] != TABLE_HEADER:
        found = f"`| {' | '.join(rows[0])} |`" if rows else "no table at all"
        offenders.append(
            f"{label}: the `{SECTION_HEADING}` table must open with `| {header} |`, "
            f"found {found}."
        )
    body = _body_rows(block)
    if len(body) < MIN_TABLE_ROWS:
        offenders.append(
            f"{label}: {len(body)} non-total rows, {MIN_TABLE_ROWS} is the minimum. "
            "A total with nothing under it is the bare number again."
        )
    return offenders


def _row_content_offenders(label: str, text: str) -> list[str]:
    block = _size_estimate_block(text)
    if block is None:
        return []
    offenders = []
    for row in _body_rows(block):
        if len(row) < len(TABLE_HEADER):
            offenders.append(f"{label}: row {row} has fewer than {len(TABLE_HEADER)} cells.")
            continue
        term, lines, source = row[0], row[1].strip("* "), row[2]
        if not _INT_RE.fullmatch(lines):
            offenders.append(
                f"{label}: row `{term}` states `{row[1]}` where a whole number of lines "
                "is required."
            )
        if len("".join(source.split())) < MIN_SOURCE_CHARS:
            offenders.append(
                f"{label}: row `{term}` names no source. Every number cites the `file:line` span "
                "or the command it came from."
            )
    return offenders


def _review_round_offenders(label: str, text: str) -> list[str]:
    block = _size_estimate_block(text)
    if block is None:
        return []
    count = sum(
        1 for row in _body_rows(block) if row[0].strip("* ").startswith(REVIEW_ROUND_TERM)
    )
    if count != 1:
        return [
            f"{label}: {count} rows whose Term begins `{REVIEW_ROUND_TERM}`, exactly 1 is "
            "required. "
            "The rounds a change's shape provokes are part of the estimate (ADR 066); a row of `0` "
            "with a stated reason is an answer, an absent row is not."
        ]
    return []


def _sum_offenders(label: str, text: str) -> list[str]:
    block = _size_estimate_block(text)
    if block is None:
        return []
    total_row = _total_row(block)
    if total_row is None or len(total_row) < 2:
        return [f"{label}: the `{SECTION_HEADING}` table has no `| **Total** | NNN | |` row."]
    total = _first_int(total_row[1])
    if total is None or total == 0:
        return [f"{label}: the Total row states `{total_row[1]}` where a whole number is required."]
    rows_total = 0
    for row in _body_rows(block):
        if len(row) > 1:
            rows_total += _first_int(row[1]) or 0
    if abs(rows_total - total) / total > ARBITRARY_SUM_TOLERANCE:
        return [
            f"{label}: rows sum to {rows_total} against a stated Total of {total}, further apart "
            f"than the {ARBITRARY_SUM_TOLERANCE:.0%} rounding tolerance allows."
        ]
    return []


def _landed_offenders(label: str, text: str) -> list[str]:
    status_match = _STATUS_RE.search(text)
    status = status_match.group(1) if status_match else ""
    if status not in LANDED_STATUSES:
        return []
    block = _size_estimate_block(text)
    if block is not None and _LANDED_RE.search(block):
        return []
    return [
        f"{label}: `Status: {status}` with no `Landed:` line inside `{SECTION_HEADING}` naming "
        "a number and `git diff --shortstat`. The landed number is what the next estimate reads."
    ]


RULES = (
    _missing_section_offenders,
    _table_shape_offenders,
    _row_content_offenders,
    _review_round_offenders,
    _sum_offenders,
    _landed_offenders,
)


def _every_rule(label: str, text: str) -> list[str]:
    """Every offender all of RULES find in one file, so a passing sample is checked whole."""
    offenders: list[str] = []
    for rule in RULES:
        offenders.extend(rule(label, text))
    return offenders


def _stated_multiplier(text: str) -> float | None:
    """The re-score multiplier stated inside CLAUDE.md's size-gate section, or None."""
    start = text.find(SIZE_GATE_HEADING)
    if start == -1:
        return None
    section = text[start:]
    end = section.find("\n### ")
    match = _RESCORE_RE.search(section if end == -1 else section[:end])
    return float(match.group(1)) if match else None


def _restated_multipliers(text: str) -> list[float]:
    """Every re-score multiplier a document restates in prose, in reading order."""
    return [float(value) for value in _RESTATED_RESCORE_RE.findall(text)]


_SOURCED_ROW = "| `backend/app/foo.py` | 60 | `foo.py:12-31` rewritten, counted twice |"
_REVIEW_ROW = "| Review rounds (ADR 066) | 40 | one pass at ~40, the shape ADR 066 measures |"
_LANDED_LINE = (
    "Landed: 336 (git diff --shortstat $(git merge-base master HEAD), 2026-09-18) — 1.00×"
)


def _sample(
    *rows: str,
    header: str = "| Term | Lines | Source |",
    total: str = "| **Total** | **100** | |",
    status: str = "planned",
    status_marker: str = "> ",
    landed: str = "",
    after_section: str = "",
) -> str:
    """A synthetic spec file carrying one `## Size estimate` section, built for one assertion.

    `landed` goes inside that section; `after_section` goes into the `## Risks` section below it.
    """
    lines = [
        "# Spec 999 — A synthetic sample",
        "",
        f"{status_marker}**Status:** {status}",
        "",
        SECTION_HEADING,
        "",
        header,
        "| --- | --- | --- |",
        *rows,
        total,
    ]
    if landed:
        lines += ["", landed]
    lines += ["", "## Risks", ""]
    if after_section:
        lines += [after_section, ""]
    return "\n".join(lines) + "\n"


def _offenders_over(tmp_path, monkeypatch, rule, text: str, name: str = "plan.md") -> list[str]:
    """Run one rule over a synthetic specs/ tree holding a single spec file."""
    directory = tmp_path / "999-synthetic"
    directory.mkdir(exist_ok=True)
    for stale in directory.iterdir():
        stale.unlink()
    (directory / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr("tests.test_spec_estimates.SPECS_DIR", tmp_path)
    offenders = []
    for path in _scanned_files():
        offenders.extend(rule(f"{path.parent.name}/{path.name}", path.read_text(encoding="utf-8")))
    return offenders


def test_every_scanned_spec_derives_its_estimate():
    offenders = []
    for path in _scanned_files():
        label = f"{path.parent.name}/{path.name}"
        offenders.extend(_every_rule(label, path.read_text(encoding="utf-8")))

    assert not offenders, "\n".join(offenders)


def test_the_scan_covers_the_spec_that_introduced_the_gate():
    """A gate whose walk finds nothing passes forever, so the walk itself is pinned."""
    scanned = _scanned_files()
    numbers = {int(_SPEC_DIR_RE.match(path.parent.name).group(1)) for path in scanned}

    assert scanned, (
        f"no specs/NNN-*/plan.md or fix.md with NNN >= {SCANNED_FROM} was found, so every other "
        "test in this module is vacuous"
    )
    assert SCANNED_FROM in numbers, (
        f"specs/{SCANNED_FROM}-* is the spec that introduced this gate and must be inside its own "
        f"scan; scanned {sorted(numbers)}"
    )


def test_a_bare_estimate_fails_and_a_table_passes(tmp_path, monkeypatch):
    bare = (
        "# Fix 999 — A synthetic sample\n\n"
        "> **Status:** planned\n\n"
        "## Score\n\n"
        "- Estimated diff **~45 lines** (`git diff --shortstat` unit); "
        "record the real number on landing.\n"
    )

    failed = _offenders_over(tmp_path, monkeypatch, _every_rule, bare, "fix.md")
    passed = _offenders_over(tmp_path, monkeypatch, _every_rule, _sample(_SOURCED_ROW, _REVIEW_ROW))

    assert failed and all(SECTION_HEADING in message for message in failed)
    assert not passed


def test_a_row_with_no_source_fails_and_a_sourced_row_passes(tmp_path, monkeypatch):
    sourceless = "| `backend/app/foo.py` | 60 |  |"
    piped = "| `backend/tests/` | 60 | `wc -l \\| sort` over the suite gave 60 |"

    failed = _offenders_over(
        tmp_path, monkeypatch, _row_content_offenders, _sample(sourceless, _REVIEW_ROW)
    )
    passed = _offenders_over(
        tmp_path, monkeypatch, _row_content_offenders, _sample(_SOURCED_ROW, _REVIEW_ROW)
    )

    assert failed and all("names no source" in message for message in failed)
    assert not passed
    assert not _offenders_over(
        tmp_path, monkeypatch, _row_content_offenders, _sample(piped, _REVIEW_ROW)
    )


def test_a_non_integer_lines_cell_fails_and_an_integer_one_passes(tmp_path, monkeypatch):
    approximate = "| `backend/app/foo.py` | ~60 | `foo.py:12-31` rewritten, counted twice |"

    failed = _offenders_over(
        tmp_path, monkeypatch, _row_content_offenders, _sample(approximate, _REVIEW_ROW)
    )
    passed = _offenders_over(
        tmp_path, monkeypatch, _row_content_offenders, _sample(_SOURCED_ROW, _REVIEW_ROW)
    )

    assert failed and all("whole number of lines" in message for message in failed)
    assert not passed


def test_a_table_with_no_review_round_row_fails_and_one_with_it_passes(tmp_path, monkeypatch):
    second_file_row = "| `backend/tests/test_foo.py` | 40 | `test_foo.py:1-40` replaced wholesale |"
    zero_round_row = "| Review rounds — 0, the diff is a constant rename | 0 | no pin is added |"

    failed = _offenders_over(
        tmp_path,
        monkeypatch,
        _review_round_offenders,
        _sample(_SOURCED_ROW, second_file_row),
    )
    passed = _offenders_over(
        tmp_path,
        monkeypatch,
        _review_round_offenders,
        _sample(_SOURCED_ROW, second_file_row, zero_round_row),
    )

    assert failed and all(REVIEW_ROUND_TERM in message for message in failed)
    assert not passed


def test_rows_that_do_not_sum_fail_and_rows_within_tolerance_pass(tmp_path, monkeypatch):
    failed = _offenders_over(
        tmp_path,
        monkeypatch,
        _sum_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, total="| **Total** | **200** | |"),
    )
    passed = _offenders_over(
        tmp_path,
        monkeypatch,
        _sum_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, total="| **Total** | **105** | |"),
    )

    assert failed and all("rows sum to 100" in message for message in failed)
    assert not passed


def test_a_landed_file_with_no_measured_line_fails_and_one_with_it_passes(tmp_path, monkeypatch):
    failed = _offenders_over(
        tmp_path,
        monkeypatch,
        _landed_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, status="done"),
    )
    outside_the_section = _offenders_over(
        tmp_path,
        monkeypatch,
        _landed_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, status="done", after_section=_LANDED_LINE),
    )
    plain_status = _offenders_over(
        tmp_path,
        monkeypatch,
        _landed_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, status="done", status_marker=""),
    )
    passed = _offenders_over(
        tmp_path,
        monkeypatch,
        _landed_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, status="done", landed=_LANDED_LINE),
    )
    plain_status_passed = _offenders_over(
        tmp_path,
        monkeypatch,
        _landed_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, status="done", status_marker="", landed=_LANDED_LINE),
    )

    assert failed and all("Landed:" in message for message in failed)
    assert outside_the_section and all(SECTION_HEADING in m for m in outside_the_section)
    assert plain_status and all("Landed:" in message for message in plain_status)
    assert not passed
    assert not plain_status_passed


def test_a_wrong_table_header_fails_and_the_standard_header_passes(tmp_path, monkeypatch):
    failed = _offenders_over(
        tmp_path,
        monkeypatch,
        _table_shape_offenders,
        _sample(_SOURCED_ROW, _REVIEW_ROW, header="| Term | Lines | Where |"),
    )
    one_row = _offenders_over(
        tmp_path, monkeypatch, _table_shape_offenders, _sample(_REVIEW_ROW)
    )
    passed = _offenders_over(
        tmp_path, monkeypatch, _table_shape_offenders, _sample(_SOURCED_ROW, _REVIEW_ROW)
    )

    assert failed and all("must open with" in message for message in failed)
    assert all("`| Term | Lines | Where |`" in message for message in failed), (
        "the header a spec actually wrote must be echoed as a markdown row, not as a repr"
    )
    assert one_row and all("non-total rows" in message for message in one_row)
    assert not passed


def test_a_near_miss_heading_is_quoted_back_and_a_bare_absence_is_not(tmp_path, monkeypatch):
    base = _sample(_SOURCED_ROW, _REVIEW_ROW)
    decorated = _offenders_over(
        tmp_path,
        monkeypatch,
        _missing_section_offenders,
        base.replace(SECTION_HEADING, f"{SECTION_HEADING} (tracked lines only)", 1),
    )
    demoted = _offenders_over(
        tmp_path,
        monkeypatch,
        _missing_section_offenders,
        base.replace(SECTION_HEADING, "### Size estimate", 1),
    )
    absent = _offenders_over(
        tmp_path,
        monkeypatch,
        _missing_section_offenders,
        base.replace(SECTION_HEADING, "## Working notes", 1),
    )
    passed = _offenders_over(tmp_path, monkeypatch, _missing_section_offenders, base)

    assert decorated and all("(tracked lines only)" in message for message in decorated)
    assert demoted and all("`### Size estimate`" in message for message in demoted)
    assert absent and all(f"no `{SECTION_HEADING}` heading" in message for message in absent)
    assert not passed


def test_claude_md_states_the_rescore_multiplier():
    """The multiplier exists once, and the prose half is compared against it rather than trusted."""
    if not CLAUDE_MD.is_file():
        pytest.skip("CLAUDE.md is gitignored and absent from clean checkouts")

    drifted_value = RESCORE_MULTIPLIER + 1
    stated = _stated_multiplier(CLAUDE_MD.read_text(encoding="utf-8"))
    drifted = _stated_multiplier(
        f"{SIZE_GATE_HEADING}\n\nIf the real diff lands more than {drifted_value:g}× the "
        "estimate, the estimate was defective.\n"
    )

    assert stated == RESCORE_MULTIPLIER, (
        f"CLAUDE.md's size-gate section states {stated}× where RESCORE_MULTIPLIER is "
        f"{RESCORE_MULTIPLIER}. One of the two is stale; ADR 080 shows the arithmetic."
    )
    assert drifted == drifted_value, (
        f"the pin must read the multiplier out of the live wording; a section stating "
        f"{drifted_value:g}× was read as {drifted}"
    )


@pytest.mark.parametrize(
    "document",
    MULTIPLIER_RESTATEMENTS,
    ids=("commands-fix", "templates-plan"),
)
def test_a_restatement_of_the_rescore_multiplier_agrees_with_the_pin(document: Path):
    """The number lives in three documents, and the two copies are read rather than trusted.

    A copy drifting from CLAUDE.md sends the same landed ratio to two verdicts
    depending on which document the reader opened.
    """
    if not document.is_file():
        pytest.skip(f"{document.name} is gitignored and absent from clean checkouts")

    drifted_value = RESCORE_MULTIPLIER + 1
    stated = _restated_multipliers(document.read_text(encoding="utf-8"))
    drifted = _restated_multipliers(
        f"if the ratio is over {drifted_value:g}× the estimate, the same line names the row"
    )

    assert stated, (
        f"{document.name} restates the re-score multiplier and this pin found none there, "
        "so the wording moved and the pin now watches nothing"
    )
    assert set(stated) == {RESCORE_MULTIPLIER}, (
        f"{document.name} states {sorted(set(stated))} where RESCORE_MULTIPLIER is "
        f"{RESCORE_MULTIPLIER}; one of the two is stale and ADR 080 shows the arithmetic"
    )
    assert drifted == [drifted_value], (
        f"the pin must read the multiplier out of the live wording; prose stating "
        f"{drifted_value:g}× was read as {drifted}"
    )
