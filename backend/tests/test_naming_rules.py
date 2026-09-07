"""`docs/style-guide.md` §2.1's suffix rule, enforced instead of documented.

The guide splits two suffixes that read alike: a `*Outcome` names a closed set
of tokens and is therefore an `Enum`, while a `*Result` carries a payload and
is therefore not. Prose cannot stop the next payload class from being called
`SomethingOutcome`; this file fails when it is.

The check reads the AST rather than importing, so a module whose import needs
an optional `[cloud]` or `[local]` extra is still covered.
"""

import ast
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}


def _suffixed_classes() -> list[tuple[str, ast.ClassDef]]:
    """Every `*Result` / `*Outcome` class declared under `app`, with its module."""
    found: list[tuple[str, ast.ClassDef]] = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        module = path.relative_to(_APP_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith(
                ("Result", "Outcome")
            ):
                found.append((module, node))
    return found


def _has_enum_base(node: ast.ClassDef) -> bool:
    """Whether any base of the class names one of the stdlib enum types."""
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name in _ENUM_BASES:
            return True
    return False


def test_outcome_classes_are_enums():
    offenders = [
        f"{module}:{node.name}"
        for module, node in _suffixed_classes()
        if node.name.endswith("Outcome") and not _has_enum_base(node)
    ]
    assert not offenders, (
        "a *Outcome class must be an Enum (style-guide §2.1); rename these to "
        f"*Result or give them an Enum base: {offenders}"
    )


def test_result_classes_are_not_enums():
    offenders = [
        f"{module}:{node.name}"
        for module, node in _suffixed_classes()
        if node.name.endswith("Result") and _has_enum_base(node)
    ]
    assert not offenders, (
        "a *Result class carries a payload and must not be an Enum "
        f"(style-guide §2.1); rename these to *Outcome: {offenders}"
    )


def test_the_walk_finds_the_declarations_it_is_meant_to_check():
    """Both rules above pass vacuously on an empty walk, so pin that it is not."""
    names = {node.name for _, node in _suffixed_classes()}
    assert {"UpdateResult", "RelocateOutcome", "ConsolidateOutcome"} <= names
