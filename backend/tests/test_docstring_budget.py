"""Docstrings under `app` state a contract and stop (ADR 077).

The cap is the outcome-shaped half: a docstring this short cannot hold a
restated argument whatever words it picks. The banned list below is an
enumeration of the vectors measured in this repository on 2026-09-18 and is
incomplete by construction; a new vector is a new entry, not a defect in this
file. Neither half checks that a docstring is *true*. An indented docstring
ends on a whitespace-only line that no reader sees, and it is not counted.
"""

import ast
import re
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_LINE_BUDGET = {"module": 8, "def": 5}

_ROT_VECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a file:line reference, which is wrong the moment that file moves",
        re.compile(r"\b[\w./]+\.(?:py|ts|rs|swift):\d+"),
    ),
    (
        "a task or spec id, and specs/ is gitignored and absent from every clone",
        re.compile(r"\bJS-\d+\b|\bspecs?\s+\d{3}\b", re.IGNORECASE),
    ),
    (
        "a count of sites or files, which is wrong the moment one is added",
        re.compile(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"(?:call sites?|sites?|files?|places?|copies|members|entries)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "narrated history, which describes a state the reader cannot see",
        re.compile(
            r"\bused to\b|\bpreviously\b|\boriginally\b"
            r"|\bthe previous implementation\b|\bprior to this\b",
            re.IGNORECASE,
        ),
    ),
    (
        "a review round, which belongs in the spec that ran it",
        re.compile(r"\breview (?:round|pass|iteration)\b|\b(?:RED|GREEN|YELLOW)-\d"),
    ),
)

_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _measured_lines(text: str) -> int:
    """The lines a reader sees, ignoring the indent left before the closing quotes."""
    lines = text.splitlines()
    return len(lines) - 1 if lines and not lines[-1].strip() else len(lines)


def _docstrings() -> list[tuple[str, str, str, str]]:
    """Every docstring under `app` as (module, symbol, kind, text)."""
    found: list[tuple[str, str, str, str]] = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        module = path.relative_to(_APP_DIR).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, _DOCUMENTED):
                continue
            text = ast.get_docstring(node, clean=False)
            if text is None:
                continue
            kind = "module" if isinstance(node, ast.Module) else "def"
            found.append((module, getattr(node, "name", "<module>"), kind, text))
    return found


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


def test_the_walk_reaches_every_package_it_is_meant_to_check():
    """Both rules pass vacuously on an empty walk, so pin that it is not."""
    packages = {module.split("/")[0] for module, _, _, _ in _docstrings() if "/" in module}
    assert {"audio", "stt", "transcripts", "core", "preferences"} <= packages
    assert len(_docstrings()) > 300
