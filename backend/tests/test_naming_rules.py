"""`docs/style-guide.md` §2.1's naming rules, enforced instead of documented.

The suffix rule splits two endings that read alike: a `*Outcome` names a closed
set of tokens and is therefore an `Enum`, while a `*Result` carries a payload
and is therefore not. Prose cannot stop the next payload class from being
called `SomethingOutcome`; this file fails when it is.

The acronym rule covers the family whose members spell `STT` and `LLM` in
capitals — `STTProvider`, `STTSettings`, `LocalSTTProvider`,
`WhisperCppServerSTTProvider`. One member carried a title-case `Stt` until
JS-178 recased it, and nothing failed while it did; this rule is what fails now.

What the acronym rule does **not** cover, deliberately: `Gpu`, `Vad` and `Api`,
a separate title-case family that §2.1 leaves alone; acronyms it does not list,
because §2.1's rule is "match the neighbours of the type you are adding", so a
new acronym joins `_ALL_CAPS_ACRONYMS` only once its own family is settled;
names that are not class declarations; and anything outside `app`, so the
TypeScript mirror of these types in `src/api.ts` is not read here. `LLM` is
listed because §2.1 lists it, and has no class under `app` today.

Both checks read the AST rather than importing, so a module whose import needs
an optional `[cloud]` or `[local]` extra is still covered.
"""

import ast
import re
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}

_ALL_CAPS_ACRONYMS = {"STT", "LLM"}

_CAMEL_SEGMENT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _declared_classes() -> list[tuple[str, ast.ClassDef]]:
    """Every class declared under `app`, with the module that declares it."""
    found: list[tuple[str, ast.ClassDef]] = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        module = path.relative_to(_APP_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.append((module, node))
    return found


def _suffixed_classes() -> list[tuple[str, ast.ClassDef]]:
    """Every `*Result` / `*Outcome` class declared under `app`, with its module."""
    return [
        (module, node)
        for module, node in _declared_classes()
        if node.name.endswith(("Result", "Outcome"))
    ]


def _has_enum_base(node: ast.ClassDef) -> bool:
    """Whether any base of the class names one of the stdlib enum types."""
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name in _ENUM_BASES:
            return True
    return False


def _miscased_acronyms(name: str) -> list[str]:
    """The CamelCase segments of `name` that spell a listed acronym in the wrong case.

    Segmenting first is what keeps `SmallModel` and `Hallmark` out of it: a
    plain case-insensitive search for `llm` matches across the word boundary
    in both, while neither yields a segment spelling the acronym.
    """
    return [
        segment
        for segment in _CAMEL_SEGMENT.findall(name)
        if segment.upper() in _ALL_CAPS_ACRONYMS and segment != segment.upper()
    ]


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


def test_stt_and_llm_acronyms_are_all_caps_in_class_names():
    offenders = [
        f"{module}:{node.name} -> {_miscased_acronyms(node.name)}"
        for module, node in _declared_classes()
        if _miscased_acronyms(node.name)
    ]
    assert not offenders, (
        "STT and LLM are all-caps in this class family (style-guide §2.1); "
        f"recase these: {offenders}"
    )


def test_the_walk_finds_the_declarations_it_is_meant_to_check():
    """Both suffix rules pass vacuously on an empty walk, so pin that it is not."""
    names = {node.name for _, node in _suffixed_classes()}
    assert {"UpdateResult", "RelocateOutcome", "ConsolidateOutcome"} <= names


def test_the_walk_finds_the_acronym_family_it_is_meant_to_check():
    """The acronym rule passes vacuously on an empty walk, so pin that it is not.

    Named here are family members this rule does not itself move, so a failure
    of the rule above stays distinguishable from the walk going blind.
    """
    names = {node.name for _, node in _declared_classes()}
    assert {"STTProvider", "STTSettings", "LocalSTTProvider"} <= names
