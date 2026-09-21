"""Docstrings under `app` state a contract and stop (ADR 077).

The budget counts lines that carry text, so blank lines — including the PEP 257
separator after the summary — are free; counting them bought an author a line of
content by deleting one. The banned list below enumerates the vectors measured in
this repository on 2026-09-18 and is incomplete by construction; a new vector is a
new entry, not a defect in this file. Neither half checks that a docstring is true.
"""

import ast
import functools
import re

from tests.app_modules import app_modules

_LINE_BUDGET = {"module": 8, "def": 5}

_ROT_VECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a path or file:line reference, wrong the moment that file moves or is renamed",
        re.compile(r"\b[\w./-]+\.(?:py|ts|rs|swift):\d+|\b(?:docs|specs)/[\w./-]+"),
    ),
    (
        "a task or spec id, and docs/ and specs/ are gitignored and absent from every clone",
        re.compile(r"\bJS-\d+\b|\bspecs?\s+\d{2,3}\b", re.IGNORECASE),
    ),
    (
        "a count of code locations, wrong the moment one is added",
        re.compile(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"(?:call sites?|sites?|places?|copies|callers?|modules?)\b"
            r"|\bin \d+ files\b",
            re.IGNORECASE,
        ),
    ),
    (
        "narrated history, describing a state the reader cannot see",
        re.compile(
            r"(?<!is )(?<!are )(?<!was )(?<!were )(?<!be )(?<!being )(?<!been )\bused to\b"
            r"|\bpreviously\b|\boriginally\b|\bthe previous implementation\b"
            r"|\bprior to this\b|\bbefore this change\b",
            re.IGNORECASE,
        ),
    ),
    (
        "a review round, which belongs in the spec that ran it",
        re.compile(r"\breview (?:round|pass|iteration)\b|\b(?:RED|GREEN|YELLOW)-\d", re.IGNORECASE),
    ),
)

_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _measured_lines(text: str) -> int:
    """The lines carrying text; blank lines and the closing indent are free."""
    return sum(1 for line in text.splitlines() if line.strip())


@functools.cache
def _docstrings() -> tuple[tuple[str, str, str, str], ...]:
    """Every docstring under `app` as (module, symbol, kind, text)."""
    found: list[tuple[str, str, str, str]] = []
    for module, path in app_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, _DOCUMENTED):
                continue
            text = ast.get_docstring(node, clean=False)
            if text is None:
                continue
            kind = "module" if isinstance(node, ast.Module) else "def"
            found.append((module, getattr(node, "name", "<module>"), kind, text))
    return tuple(found)


def test_every_docstring_fits_the_line_budget_for_its_kind():
    over = [
        f"{module}:{symbol} is {_measured_lines(text)} lines, budget {_LINE_BUDGET[kind]}"
        for module, symbol, kind, text in _docstrings()
        if _measured_lines(text) > _LINE_BUDGET[kind]
    ]
    assert not over, (
        "a docstring states the contract and stops (ADR 077); move the reasoning "
        f"into an ADR and cut these back: {over}"
    )


def test_no_docstring_carries_a_shape_that_goes_stale_on_its_own():
    offenders = [
        f"{module}:{symbol} carries {why} -- {match.group(0)!r}"
        for module, symbol, _, text in _docstrings()
        for why, pattern in _ROT_VECTORS
        if (match := pattern.search(text))
    ]
    assert not offenders, (
        "these docstrings make a claim that a change elsewhere silently falsifies "
        f"(ADR 077): {offenders}"
    )


def test_each_rot_vector_separates_the_shape_it_bans_from_ordinary_contract_prose():
    """Three of these matched ordinary prose or missed the banned shape when written."""
    banned = (
        "Previously this lived elsewhere.",
        "Review round 3 found this.",
        "a review pass changed it",
        "JS-186 decided it.",
        "spec 29 decided it.",
        "See specs/186-docstrings-grow-faster-than-code/plan.md",
        "Described in docs/adr/042-a-thing.md",
        "Inlined at app/stt/routing.py:140",
        "Read at seven call sites.",
        "It used to return None.",
    )
    allowed = (
        "`dtype` is used to interpret the buffer.",
        "Assembly writes two files at the target rate.",
        "The copy writes all ten columns.",
        "Returns two values, the provider and the reason.",
        "See ADR 073 for the decision.",
        "Reviews the header before reading blocks.",
    )
    missed = [s for s in banned if not any(p.search(s) for _, p in _ROT_VECTORS)]
    flagged = [s for s in allowed if any(p.search(s) for _, p in _ROT_VECTORS)]
    assert not missed, f"these banned shapes walk straight past the gate: {missed}"
    assert not flagged, f"these are ordinary contract prose and must not be flagged: {flagged}"


def test_the_walk_reaches_every_package_it_is_meant_to_check():
    """Every rule above passes vacuously on an empty walk, so pin that it is not."""
    packages = {module.split("/")[0] for module, _, _, _ in _docstrings() if "/" in module}
    swept = {"audio", "stt", "transcripts", "core", "preferences", "embeddings", "pipeline"}
    assert swept <= packages
    assert len(_docstrings()) > 300
