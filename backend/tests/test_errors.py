"""The hierarchy's shape, pinned where prose cannot hold it.

Six properties live here: the base is `Exception` and not `RuntimeError`,
the three subclasses are members of it, the base itself cannot be raised, no
subclass answers the base's 500 sentinel, and no two subclasses share a `code`.
The last two are what a step-2 migrator relies on when adding a fourth class —
a forgotten `status_code` override and a copy-pasted `code` both turn red here
rather than in a response nobody inspects. The third closes the shortcut those
two leave open: a bare `JustSayError` would otherwise answer the sentinel and
look like a refusal at 500.

Mutations actually run against `app/core/errors.py`, with the number of tests
each one reddens across this file and `tests/test_error_handler.py` together:

- `JustSayError` re-based on `RuntimeError` -- four tests
- `status_code` dropped from `NotReadyError` -- two tests, since the class then
  answers the sentinel here and a 500 there
- `ResourceUnavailableError.code` set to `"configuration_error"` -- four tests
- `from fastapi import HTTPException` planted in the module -- two tests, the
  AST one and the `sys.modules` subprocess one
- the `type(self) is JustSayError` guard dropped from `__init__` -- two tests,
  one here and one in `tests/test_error_handler.py`

The sixth property is the repo-wide one, and it is the only thing holding a
hierarchy whose classes are declared in the package that raises them rather
than in `app/core/errors.py` (ADR 060). It imports the modules the source walk
below found an exception class in and then walks every subclass reachable from
`JustSayError`, so a package-local class is covered by exactly the two rules
the module-scoped tests above apply to the three base ones, and the two halves
describe the same set by construction rather than by whatever else the test
process happened to import.
Mutation run: `SessionMismatchError.code` set to `"not_ready"` -- one test.
"""

import ast
import builtins
import functools
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import errors
from app.core.errors import (
    ConfigurationError,
    JustSayError,
    NotReadyError,
    ResourceUnavailableError,
)

_ERRORS_SOURCE = Path(errors.__file__)
_APP_DIR = Path(__file__).resolve().parents[1] / "app"
_DELIBERATELY_OUTSIDE_THE_HIERARCHY = frozenset(
    {"app.audio.analysis.MalformedCaptureBlockError"}
)
_BUILTIN_EXCEPTION_BASES = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)
_DECLARED_EXCEPTION_CLASS_COUNT = 14
_HIERARCHY_MEMBER_COUNT = 13
_WEB_FRAMEWORK_ROOTS = frozenset({"fastapi", "starlette"})


def _declared_subclasses() -> list[type[JustSayError]]:
    """Every strict subclass of `JustSayError` reachable from the module."""
    return [
        member
        for member in vars(errors).values()
        if isinstance(member, type)
        and issubclass(member, JustSayError)
        and member is not JustSayError
    ]


def test_the_base_is_an_exception_and_not_a_runtime_error() -> None:
    assert JustSayError.__bases__ == (Exception,)
    assert not issubclass(JustSayError, RuntimeError)


@pytest.mark.parametrize(
    "subclass", [ConfigurationError, ResourceUnavailableError, NotReadyError]
)
def test_each_named_subclass_belongs_to_the_hierarchy(subclass: type) -> None:
    assert issubclass(subclass, JustSayError)
    assert not issubclass(subclass, RuntimeError)


def test_the_three_named_subclasses_are_all_there_are_today() -> None:
    assert set(_declared_subclasses()) == {
        ConfigurationError,
        ResourceUnavailableError,
        NotReadyError,
    }


def test_no_subclass_resolves_to_the_base_s_500_sentinel() -> None:
    """500 means "a crash", so a refusal answering it is a contradiction."""
    assert JustSayError.status_code == 500
    offenders = [c.__name__ for c in _declared_subclasses() if c.status_code == 500]
    assert offenders == []


def test_the_base_cannot_be_raised_as_a_refusal_of_its_own() -> None:
    """Membership is what the base is for; answering a request is not."""
    with pytest.raises(TypeError, match="membership test"):
        JustSayError("Something went wrong internally.")


def test_every_named_subclass_stays_constructible() -> None:
    for subclass in _declared_subclasses():
        assert isinstance(subclass("A refusal."), JustSayError)


def test_configuration_error_keeps_the_code_the_widget_branches_on() -> None:
    """The other half of this literal lives in `src/api.ts`.

    `CONFIGURATION_ERROR_CODE` there is what `src/widget/error-label.ts` tests
    to choose the "Add key in Settings" label, and nothing mechanically
    compares the two strings across the language boundary. Renaming this one
    turns this test red; renaming the TypeScript one alone leaves both suites
    green and silently retires that label, which is why a reader who breaks
    this is told here where the other half is.
    """
    assert ConfigurationError.code == "configuration_error"


def test_every_subclass_declares_a_unique_code() -> None:
    codes = [c.code for c in _declared_subclasses()]
    assert len(codes) == len(set(codes))
    assert JustSayError.code not in codes


def test_the_constructor_keeps_the_message_the_diagnostic_and_the_headers_apart() -> None:
    exc = ConfigurationError(
        "Add your Gemini API key in Settings.",
        diagnostic="provider replied 401 invalid_api_key",
        headers={"Retry-After": "1"},
    )
    assert str(exc) == "Add your Gemini API key in Settings."
    assert exc.message == "Add your Gemini API key in Settings."
    assert exc.diagnostic == "provider replied 401 invalid_api_key"
    assert exc.headers == {"Retry-After": "1"}


def test_the_diagnostic_and_the_headers_are_optional() -> None:
    exc = NotReadyError("Nothing is recording.")
    assert exc.diagnostic is None
    assert exc.headers is None


def test_the_errors_module_imports_no_web_framework() -> None:
    """Every layer must be able to raise from here, so nothing web-shaped."""
    tree = ast.parse(_ERRORS_SOURCE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    assert roots & _WEB_FRAMEWORK_ROOTS == set()


def test_importing_the_module_does_not_pull_a_web_framework_into_the_process() -> None:
    """The AST check misses a transitive import; this one does not."""
    code = (
        "import sys, app.core.errors; "
        "assert 'fastapi' not in sys.modules and 'starlette' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ERRORS_SOURCE.parents[2]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _every_subclass() -> list[type[JustSayError]]:
    """Every subclass in the process, however deep and wherever declared.

    The walk supplies its own imports, because a class in a module nothing has
    imported does not exist yet and `__subclasses__()` cannot reach it. Being
    reached by `app.main` is not the same thing: after importing it alone the
    walk finds 11 of the 14 declarations, and
    `app.transcripts.vector_store.SemanticSearchUnavailableError` arrived only
    because an autouse fixture in `tests/conftest.py` imports that module for
    an unrelated reason. `pytest --noconftest tests/test_errors.py` therefore
    failed here, naming a class that had never left the hierarchy.

    The modules imported are the ones the source walk below found a class in,
    so the runtime half and the source half describe the same set by
    construction rather than by whatever else the process happened to load.
    """
    for module in sorted(
        name.rsplit(".", 1)[0] for name in _declared_exception_class_names()
    ):
        importlib.import_module(module)

    found: list[type[JustSayError]] = []
    pending = [JustSayError]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in found:
                found.append(subclass)
                pending.append(subclass)
    return found


def test_no_refusal_anywhere_in_the_app_answers_the_500_sentinel() -> None:
    """A `JustSayError` that answers 500 is indistinguishable from the crash
    the hierarchy exists to separate it from."""
    offenders = sorted(
        f"{c.__module__}.{c.__name__}" for c in _every_subclass() if c.status_code == 500
    )
    assert offenders == []


def test_every_refusal_that_declares_a_code_declares_a_distinct_one() -> None:
    """Inheriting a code says "the same kind of refusal", which is legitimate;
    two classes independently declaring one value is a collision the frontend
    would have no way to tell apart.
    """
    declared = [c for c in _every_subclass() if "code" in c.__dict__]
    codes = [c.code for c in declared]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    assert duplicates == []
    assert JustSayError.code not in codes
    assert len(declared) >= 3


def _dotted_name(expression: ast.expr) -> str:
    """A base expression as the source spells it, or "" if it is not a name."""
    parts: list[str] = []
    while isinstance(expression, ast.Attribute):
        parts.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name):
        return ""
    parts.append(expression.id)
    return ".".join(reversed(parts))


def _module_and_package(path: Path) -> tuple[str, str]:
    """What `cls.__module__` says for a class in `path`, and its package.

    `__init__.py` is dropped rather than kept as a part. A class declared in
    `app/x/__init__.py` reports `app.x` at runtime while the file path spells
    `app.x.__init__`, so keeping it made that class a permanent stray: present
    in the source walk under one name and in the runtime walk under another.
    The package is what a relative import counts back from, and for a package
    `__init__` that is the package itself.
    """
    parts = path.relative_to(_APP_DIR.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), ".".join(parts[:-1])
    return ".".join(parts), ".".join(parts[:-1])


def _import_aliases(tree: ast.Module, package: str) -> dict[str, str]:
    """Every name this module's imports bind, mapped to what it names elsewhere.

    `from app.core.errors import NotReadyError as Base` binds `Base` to
    `app.core.errors.NotReadyError`, and `import ctypes` binds `ctypes` to
    itself. Relative imports are counted back from `package`.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    aliases[head] = head
        elif isinstance(node, ast.ImportFrom):
            root = package
            if node.level:
                root = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1])
            source = ".".join(part for part in (root if node.level else "", node.module) if part)
            for alias in node.names:
                aliases[alias.asname or alias.name] = (
                    f"{source}.{alias.name}" if source else alias.name
                )
    return aliases


def _declared_classes_in(source: str, module: str, package: str) -> dict[str, list[str]]:
    """Every class `source` declares at module level, by its qualified bases.

    Module level rather than `ast.walk`: a class declared inside a function
    body or under `if TYPE_CHECKING:` has no runtime counterpart any
    `__subclasses__()` walk could find, so counting it would fail this test
    over a class that does not exist at runtime.

    Bases are resolved through this module's own imports rather than by their
    trailing name against a repo-wide index. The index answered `Timeout` with
    whatever else in the repository happened to be called `Timeout`, so
    `class Budget(Timeout)` over a third-party class read as an app exception,
    while `from app.core.errors import NotReadyError as Base` followed by
    `class Refused(Base)` was missed entirely -- the two failures a name
    lookup produces, one in each direction. A name the imports do not bind is
    either declared here, and is qualified with `module`, or is a builtin and
    is left as it was written.
    """
    tree = ast.parse(source)
    aliases = _import_aliases(tree, package)
    local = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    def _qualified(base: ast.expr) -> str:
        dotted = _dotted_name(base)
        head, _, attribute = dotted.partition(".")
        if head in aliases:
            target = aliases[head]
            return f"{target}.{attribute}" if attribute else target
        if not attribute and head in local:
            return f"{module}.{head}"
        return dotted

    return {
        f"{module}.{node.name}": [_qualified(base) for base in node.bases]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


@functools.cache
def _module_level_classes() -> dict[str, list[str]]:
    """Every class `backend/app/` declares at module level, by its bases.

    Cached because `backend/app/` is rglob'd and parsed in full to build it,
    and four tests in this module ask for it. Every caller reads and none
    writes, which is what makes one shared answer safe to hand out.
    """
    declarations: dict[str, list[str]] = {}
    for path in sorted(_APP_DIR.rglob("*.py")):
        module, package = _module_and_package(path)
        declarations.update(
            _declared_classes_in(path.read_text(encoding="utf-8"), module, package)
        )
    return declarations


def _exception_class_names(declarations: dict[str, list[str]]) -> set[str]:
    """Which of `declarations` are exception classes, by what they derive from.

    Membership is decided by what a class derives from, not by its name ending
    in `Error`: `class CaptureRefused(Exception)` is exactly the declaration a
    naming convention hides, and the counts §3.1 states are about the
    hierarchy rather than about a spelling. In-app bases are qualified by
    `_declared_classes_in` and resolved to a fixpoint, so a class three levels
    down from `Exception` is found however the intermediate classes are named
    and wherever they were imported from.

    Every builtin exception counts as a root, not just `Exception` and
    `BaseException`. `class CaptureRefused(ValueError)` is an exception class
    by every rule §3.1 states, and a two-name root set left it invisible here
    -- all three assertions below stayed green for the exact drift this gate
    exists to catch.
    """

    def _derives_from_an_exception(bases: list[str]) -> bool:
        return any(
            base in _BUILTIN_EXCEPTION_BASES or base in exceptions for base in bases
        )

    exceptions: set[str] = set()
    growing = True
    while growing:
        found = {
            qualified
            for qualified, bases in declarations.items()
            if qualified not in exceptions and _derives_from_an_exception(bases)
        }
        exceptions |= found
        growing = bool(found)
    return exceptions


def _declared_exception_class_names() -> set[str]:
    """Every exception class `backend/app/` declares, read from the source.

    A source walk rather than a runtime one because the question is which
    classes exist, and a class outside the hierarchy is reachable from no
    `__subclasses__()` chain the runtime walk above can follow.
    """
    return _exception_class_names(_module_level_classes())


def test_a_class_deriving_from_a_builtin_other_than_exception_is_still_one() -> None:
    """`class CaptureRefused(ValueError)` is an exception, and was invisible here.

    The root set was `Exception` and `BaseException` alone, so a declaration
    rooted anywhere else in the builtin tree -- `ValueError`, `OSError`,
    `RuntimeError` -- counted as an ordinary class. The stray set, the declared
    count and the member count all stayed green while `backend/app/` grew an
    exception outside the hierarchy, which is the one drift this gate is for.
    """
    declarations = _declared_classes_in(
        "class CaptureRefused(ValueError):\n    pass\n"
        "class Louder(CaptureRefused):\n    pass\n"
        "class Settings(dict):\n    pass\n",
        "app.demo",
        "app",
    )

    assert _exception_class_names(declarations) == {
        "app.demo.CaptureRefused",
        "app.demo.Louder",
    }


def test_every_declared_error_is_in_the_hierarchy_or_named_as_staying_out() -> None:
    """A class leaving the hierarchy unnoticed is how the count went stale.

    `docs/style-guide.md` §3.1 states how many exception classes `backend/app/`
    declares and how many are members, and it named `tests/test_words.py` as
    the pin -- a module that does not mention the hierarchy at all, so the
    number was held by the paragraph asserting it. Staying out is legitimate
    and §3.1 says which cases qualify: "an invariant that broke, a library
    that misbehaved, a device that failed mid-use" keep propagating into a
    500 rather than becoming a refusal. What this rejects is drifting out in
    silence, so a class that belongs outside is added here and nowhere else.

    Both numbers are asserted, not just the set of strays: a fifteenth class
    deriving from `NotReadyError` is a legitimate member and would leave the
    stray set empty while §3.1's "declares 14 ... 13 are inside" went stale --
    the identical failure this test exists to close, relocated rather than
    fixed.
    """
    declared = _declared_exception_class_names()
    inside = {f"{c.__module__}.{c.__name__}" for c in _every_subclass()}
    inside.add(f"{JustSayError.__module__}.{JustSayError.__name__}")

    assert sorted(declared - inside) == sorted(_DELIBERATELY_OUTSIDE_THE_HIERARCHY)
    assert len(declared) == _DECLARED_EXCEPTION_CLASS_COUNT
    assert len(declared & inside) == _HIERARCHY_MEMBER_COUNT


def test_a_class_in_a_package_init_is_named_the_way_the_runtime_names_it() -> None:
    """The source walk and the runtime walk compare qualified names.

    A class in `app/x/__init__.py` answers `app.x` for `__module__`, so a
    source walk spelling it `app.x.__init__` reports a class that never left
    the hierarchy as a permanent stray -- and one that did leave it as
    accounted for.
    """
    assert _module_and_package(_APP_DIR / "audio" / "__init__.py") == (
        "app.audio",
        "app.audio",
    )
    assert _module_and_package(_APP_DIR / "audio" / "analysis.py") == (
        "app.audio.analysis",
        "app.audio",
    )


def test_a_base_is_resolved_through_the_module_s_own_imports() -> None:
    """Both directions a trailing-name lookup gets wrong.

    Judging a base by its last component against a repo-wide index calls
    `class Budget(Timeout)` an app exception because something unrelated in
    the repository is named `Timeout`, and misses
    `from app.core.errors import NotReadyError as Base` entirely, because
    `Base` is not the name of anything.
    """
    source = (
        "from app.core.errors import NotReadyError as Base\n"
        "from httpx import Timeout\n"
        "class Refused(Base):\n    pass\n"
        "class Budget(Timeout):\n    pass\n"
        "class Nested(Refused):\n    pass\n"
    )

    assert _declared_classes_in(source, "app.demo", "app") == {
        "app.demo.Refused": ["app.core.errors.NotReadyError"],
        "app.demo.Budget": ["httpx.Timeout"],
        "app.demo.Nested": ["app.demo.Refused"],
    }


def test_a_relative_import_is_resolved_against_the_declaring_package() -> None:
    """`from .errors import ...` inside `app/core/` is `app.core.errors`."""
    source = "from .errors import JustSayError\nclass Refused(JustSayError):\n    pass\n"

    assert _declared_classes_in(source, "app.core.thing", "app.core") == {
        "app.core.thing.Refused": ["app.core.errors.JustSayError"]
    }
