"""`docs/style-guide.md` §2.1's naming rules, enforced instead of documented.

The suffix rule splits two endings that read alike: a `*Outcome` names a closed
set of tokens and is therefore an `Enum`, while a `*Result` carries a payload
and is therefore not. Prose cannot stop the next payload class from being
called `SomethingOutcome`; this file fails when it is. It is a Python rule and
is read over `app` only.

The acronym rule covers the family whose members spell `STT` and `LLM` in
capitals — `STTProvider`, `STTSettings`, `LocalSTTProvider`,
`WhisperCppServerSTTProvider` — and leaves `Gpu`, `Vad` and `Api` alone, the
separate title-case family §2.1 names. It reads both declarations of a wire
type, because §7.1 makes the TypeScript interface mirror the Pydantic class
name exactly and neither declaration knows the other exists.

Python is read through the AST rather than imported, so a module whose import
needs an optional `[cloud]` or `[local]` extra is still covered. TypeScript is
read as text, the shape `test_cross_language_contracts.py` uses, so this needs
no compiler.
"""

import ast
import functools
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_APP_DIR = _REPO_ROOT / "backend" / "app"

_SRC_DIR = _REPO_ROOT / "src"

_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}

_ALL_CAPS_ACRONYMS = {"STT", "LLM"}

_CAMEL_SEGMENT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

_TS_TYPE_DECLARATION = re.compile(
    r"^(?:export )?(?:interface|type) ([A-Za-z0-9_]+)\b", re.MULTILINE
)


@functools.cache
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


@functools.cache
def _declared_typescript_types() -> list[tuple[str, str]]:
    """Every `interface` / `type` name declared under `src`, with its file.

    A declaration starts its own line, which is what separates it from an
    `import { type Thing }` specifier naming a type declared elsewhere.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(_SRC_DIR.rglob("*.ts")):
        source = path.relative_to(_REPO_ROOT).as_posix()
        for name in _TS_TYPE_DECLARATION.findall(path.read_text(encoding="utf-8")):
            found.append((source, name))
    return found


def _declared_type_names() -> list[tuple[str, str]]:
    """Every type name declared in either language, with the file declaring it."""
    return [(f"app/{module}", node.name) for module, node in _declared_classes()] + list(
        _declared_typescript_types()
    )


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


def _miscased_declarations() -> list[str]:
    """Every declared type whose name miscases a listed acronym, reported once."""
    reports: list[str] = []
    for source, name in _declared_type_names():
        segments = _miscased_acronyms(name)
        if segments:
            reports.append(f"{source}:{name} -> {segments}")
    return reports


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


def test_the_acronym_predicate_reports_a_title_case_listed_acronym():
    assert _miscased_acronyms("LocalSttStatus") == ["Stt"]
    assert _miscased_acronyms("LlmSettings") == ["Llm"]


def test_the_acronym_predicate_ignores_correct_casing_and_the_other_family():
    assert _miscased_acronyms("LocalSTTStatus") == []
    assert _miscased_acronyms("GpuVendor") == []


def test_the_acronym_predicate_ignores_words_that_merely_contain_the_letters():
    assert _miscased_acronyms("SmallModel") == []
    assert _miscased_acronyms("Hallmark") == []


def test_stt_and_llm_acronyms_are_all_caps_in_type_names():
    offenders = _miscased_declarations()
    assert not offenders, (
        "STT and LLM are all-caps in this type family (style-guide §2.1); "
        f"recase these: {offenders}"
    )


def test_the_walk_finds_the_declarations_it_is_meant_to_check():
    """Both suffix rules pass vacuously on an empty walk, so pin that it is not."""
    names = {node.name for _, node in _suffixed_classes()}
    assert {"UpdateResult", "RelocateOutcome", "ConsolidateOutcome"} <= names


def test_the_walk_finds_the_acronym_family_it_is_meant_to_check():
    """The acronym rule passes vacuously on an empty walk, so pin that it is not.

    Named alongside the type JS-178 recased are family members this rule does
    not itself move, so a failure of the rule above stays distinguishable from
    the walk going blind.
    """
    names = {node.name for _, node in _declared_classes()}
    assert {"STTProvider", "STTSettings", "LocalSTTProvider", "LocalSTTStatus"} <= names


def test_the_typescript_walk_finds_the_mirrored_types_it_is_meant_to_check():
    """The acronym rule reaches `src` only while this walk finds declarations."""
    names = {name for _, name in _declared_typescript_types()}
    assert {"LocalSTTStatus", "UserSettings", "MeetingStatus"} <= names
