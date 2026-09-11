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
than in `app/core/errors.py` (ADR 060). It walks every subclass reachable
after `app.main` is imported, so a package-local class is covered by exactly
the two rules the module-scoped tests above apply to the three base ones.
Mutation run: `SessionMismatchError.code` set to `"not_ready"` -- one test.
"""

import ast
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

    `app.main` is imported first because a class in a module nothing has
    imported does not exist yet: the composition root is what makes the walk
    repo-wide rather than errors-module-wide, and `tests/conftest.py` already
    imports it for every other test in the suite.
    """
    import app.main  # noqa: F401

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
