"""`latched_load_error` and the latch every local provider writes through it.

The frontend reads a falsy `last_error` as no error at all, so a failed load
would draw a healthy indicator over it (ADR 078). Two halves: the helper is
checked against failures carrying no message, and an AST walk over `app/stt/`
pins that no provider writes the latch any other way — which is the half that
catches a provider added later assigning `str(exc)` straight into it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.stt import base
from app.stt.base import LOAD_FAILED_WITHOUT_A_MESSAGE, latched_load_error

_STT_DIR = Path(base.__file__).resolve().parent
_LATCH_ATTRIBUTE = "_last_load_error"
_LATCH_HELPER = latched_load_error.__name__
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_PROVIDER_MODULES = frozenset({"local.py", "local_whisper_cpp.py"})


class _SilentLoadError(Exception):
    """What a bare `raise SomeError()` inside a third-party loader gives."""


def _is_helper_call(value: ast.AST) -> bool:
    """Whether an expression is a direct call to the latch helper."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id == _LATCH_HELPER
    return isinstance(func, ast.Attribute) and func.attr == _LATCH_HELPER


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node belonging to one name scope, nested function bodies excluded.

    A name bound in a sibling method must not vouch for an assignment in this
    one, so the walk stops at each nested `def` rather than flattening them.
    """
    own: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, _FUNCTIONS):
            continue
        own.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return own


def _latch_targets(node: ast.AST) -> list[ast.AST]:
    """The assigned value of `node`, once per target naming the latch attribute."""
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    else:
        return []
    return [
        value
        for target in targets
        if isinstance(target, ast.Attribute) and target.attr == _LATCH_ATTRIBUTE
    ]


def _latch_assignments() -> list[tuple[str, int, str, bool]]:
    """Every write to the latch under `app/stt/`, as (module, line, source, accepted)."""
    found: list[tuple[str, int, str, bool]] = []
    for path in sorted(_STT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scopes: list[ast.AST] = [tree]
        scopes.extend(node for node in ast.walk(tree) if isinstance(node, _FUNCTIONS))
        for scope in scopes:
            nodes = _own_nodes(scope)
            through_the_helper = {
                target.id
                for node in nodes
                if isinstance(node, ast.Assign) and _is_helper_call(node.value)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in nodes:
                for value in _latch_targets(node):
                    accepted = (
                        (isinstance(value, ast.Constant) and value.value is None)
                        or _is_helper_call(value)
                        or (isinstance(value, ast.Name) and value.id in through_the_helper)
                    )
                    found.append((path.name, node.lineno, ast.unparse(value), accepted))
    assert found, (
        f"no assignment to `{_LATCH_ATTRIBUTE}` was found under {_STT_DIR.name}/, so this "
        "walk pins nothing at all; the latch was renamed, moved or deleted"
    )
    return found


def test_a_failure_with_no_message_latches_the_fallback_sentence() -> None:
    assert latched_load_error(_SilentLoadError()) == LOAD_FAILED_WITHOUT_A_MESSAGE


def test_no_failure_shape_latches_an_error_the_widget_would_read_as_healthy() -> None:
    silent = [_SilentLoadError(), _SilentLoadError(""), OSError(), ValueError("")]
    falsy = [type(exc).__name__ for exc in silent if not latched_load_error(exc)]
    assert falsy == [], (
        f"these failures latch a falsy `last_error`, which the status widget reads as no "
        f"error and draws a healthy indicator over a failed load: {falsy}"
    )


def test_a_failure_carrying_a_message_latches_that_message_rather_than_its_class() -> None:
    latched = latched_load_error(RuntimeError("The model file is missing."))
    assert latched == "The model file is missing."


def test_every_provider_writes_the_latch_through_the_helper_or_clears_it() -> None:
    offenders = [
        f"{module}:{line} assigns `{source}`"
        for module, line, source, accepted in _latch_assignments()
        if not accepted
    ]
    assert offenders == [], (
        f"a load-error latch is `None` or a non-empty string and nothing else (ADR 078); "
        f"route these through `{_LATCH_HELPER}`: {offenders}"
    )


def test_the_latch_walk_reaches_both_local_providers() -> None:
    reached = {module for module, _, _, _ in _latch_assignments()}
    assert _PROVIDER_MODULES <= reached, (
        f"the walk covers {sorted(reached)}, which misses "
        f"{sorted(_PROVIDER_MODULES - reached)}; a provider outside it is unchecked"
    )
