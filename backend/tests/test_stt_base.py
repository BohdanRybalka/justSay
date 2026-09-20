"""`latched_load_error` and the provider attribute every local provider latches.

The frontend reads a falsy `last_error` as no error and a blank one as a toast
with no text in it (ADR 083). Two halves: the helper is checked against failures
whose message is missing or only whitespace, and an AST walk over `app/stt/`
pins every store to the provider attribute `_last_load_error` -- plain,
annotated, augmented, unpacked, looped or through `setattr` -- to `None`, a call
to the helper, or a name its own scope binds exactly once to such a call. A
store it cannot resolve is reported, never skipped. A module-level latch outside
that attribute is pinned at the published boundary instead, by
`test_local_setup.py`.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

from app.stt import base
from app.stt.base import LOAD_FAILED_WITHOUT_A_MESSAGE, latched_load_error

_STT_DIR = Path(base.__file__).resolve().parent
_LATCH_ATTRIBUTE = "_last_load_error"
_LATCH_HELPER = latched_load_error.__name__
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_PROVIDER_MODULES = frozenset({"local.py", "local_whisper_cpp.py"})
_UNRESOLVED = "a store shape this walk cannot resolve"
_DYNAMIC_SETATTR = "a setattr() naming its attribute at runtime"
_CLASS_BODY = "class P:\n    def load(self, e):\n{}\n"


class _SilentLoadError(Exception):
    """What a bare `raise SomeError()` inside a third-party loader gives."""


def _is_helper_call(value: ast.AST | None) -> bool:
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


def _paired(target: ast.AST, value: ast.AST | None) -> list[tuple[ast.AST, ast.AST | None]]:
    """Each leaf of an assignment target beside the expression it receives.

    A sequence target unpacked from a same-length sequence literal resolves per
    leaf; every other unpacking yields `None`, which no rule below accepts.
    """
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
            return [
                pair
                for element, part in zip(target.elts, value.elts)
                for pair in _paired(element, part)
            ]
        return [pair for element in target.elts for pair in _paired(element, None)]
    if isinstance(target, ast.Starred):
        return _paired(target.value, None)
    return [(target, value)]


def _stores(nodes: list[ast.AST]) -> list[tuple[ast.AST, ast.AST | None]]:
    """Every name or attribute one scope stores into, beside what it receives.

    Assignment, annotation, augmentation, `for` and `with ... as` all bind a
    name, and only the first two carry an expression the walk can read; the
    rest resolve to `None`, which no rule below accepts.
    """
    bound: list[tuple[ast.AST, ast.AST | None]] = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bound.extend(_paired(target, node.value))
        elif isinstance(node, ast.AnnAssign):
            bound.extend(_paired(node.target, node.value))
        elif isinstance(node, ast.AugAssign):
            bound.extend(_paired(node.target, None))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bound.extend(_paired(node.target, None))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bound.extend(_paired(item.optional_vars, None))
    return bound


def _setattr_calls(nodes: list[ast.AST]) -> list[tuple[ast.AST, ast.AST | None]]:
    """Every `setattr` call in one scope that can reach the latch, and its value.

    A call naming its attribute at runtime is returned unresolved rather than
    ignored, because the walk cannot tell whether that name is the latch.
    """
    calls: list[tuple[ast.AST, ast.AST | None]] = []
    for node in nodes:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "setattr" or len(node.args) != 3:
            continue
        attribute = node.args[1]
        if not isinstance(attribute, ast.Constant):
            calls.append((node, None))
        elif attribute.value == _LATCH_ATTRIBUTE:
            calls.append((node, node.args[2]))
    return calls


def _bound_once_through_the_helper(nodes: list[ast.AST]) -> frozenset[str]:
    """Names the scope binds exactly once, and binds to a direct helper call.

    A name bound twice cannot vouch for the latch whatever the order of the two
    statements, because the walk reads a scope as a set rather than in sequence.
    """
    bindings: dict[str, list[ast.AST | None]] = {}
    for node in nodes:
        if isinstance(node, ast.arg):
            bindings.setdefault(node.arg, []).append(None)
    for target, value in _stores(nodes):
        if isinstance(target, ast.Name):
            bindings.setdefault(target.id, []).append(value)
    return frozenset(
        name
        for name, values in bindings.items()
        if len(values) == 1 and _is_helper_call(values[0])
    )


def _accepted(value: ast.AST | None, through_the_helper: frozenset[str]) -> bool:
    """Whether a value written into the latch is `None` or a sentence that is not blank."""
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if _is_helper_call(value):
        return True
    return isinstance(value, ast.Name) and value.id in through_the_helper


def _latch_writes_in(label: str, tree: ast.Module) -> list[tuple[str, int, str, bool]]:
    """One module's writes to the latch, as (module, line, source, accepted).

    A store resolving to no expression and a store no scope reached are both
    carried as unaccepted, so widening the walk is the only way to silence one.
    """
    found: list[tuple[str, int, str, bool]] = []
    scopes: list[ast.AST] = [tree]
    scopes.extend(node for node in ast.walk(tree) if isinstance(node, _FUNCTIONS))
    reached: set[int] = set()
    for scope in scopes:
        nodes = _own_nodes(scope)
        through_the_helper = _bound_once_through_the_helper(nodes)
        latched = [
            (target, value)
            for target, value in _stores(nodes)
            if isinstance(target, ast.Attribute) and target.attr == _LATCH_ATTRIBUTE
        ]
        for target, value in latched + _setattr_calls(nodes):
            reached.add(id(target))
            unresolved = _DYNAMIC_SETATTR if isinstance(target, ast.Call) else _UNRESOLVED
            found.append(
                (
                    label,
                    target.lineno,
                    unresolved if value is None else ast.unparse(value),
                    value is not None and _accepted(value, through_the_helper),
                )
            )
    found.extend(
        (label, node.lineno, _UNRESOLVED, False)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == _LATCH_ATTRIBUTE
        and isinstance(node.ctx, ast.Store)
        and id(node) not in reached
    )
    return found


@functools.cache
def _latch_assignments(directory: Path = _STT_DIR) -> tuple[tuple[str, int, str, bool], ...]:
    """Every write to the latch under a speech package, subpackages included."""
    found = [
        write
        for path in sorted(directory.rglob("*.py"))
        if "__pycache__" not in path.parts
        for write in _latch_writes_in(
            path.relative_to(directory).as_posix(),
            ast.parse(path.read_text(encoding="utf-8")),
        )
    ]
    assert found, (
        f"no assignment to `{_LATCH_ATTRIBUTE}` was found under {directory.name}/, so this "
        "walk pins nothing at all; the latch was renamed, moved or deleted"
    )
    return tuple(found)


def _classified(source: str) -> list[tuple[str, bool]]:
    """Each latch write in a snippet of source, as (unparsed value, accepted)."""
    return [
        (written, accepted)
        for _, _, written, accepted in _latch_writes_in("snippet", ast.parse(source))
    ]


def test_a_failure_with_no_message_latches_the_fallback_sentence() -> None:
    assert latched_load_error(_SilentLoadError()) == LOAD_FAILED_WITHOUT_A_MESSAGE


def test_a_failure_carrying_a_message_latches_that_message_rather_than_its_class() -> None:
    latched = latched_load_error(RuntimeError("The model file is missing."))
    assert latched == "The model file is missing."


def test_a_whitespace_only_failure_message_latches_the_fallback_sentence() -> None:
    assert latched_load_error(RuntimeError("   ")) == LOAD_FAILED_WITHOUT_A_MESSAGE


def test_no_failure_shape_latches_a_message_the_toast_would_render_empty() -> None:
    blank = [
        _SilentLoadError(),
        _SilentLoadError(""),
        _SilentLoadError(" "),
        OSError(),
        OSError("\n"),
        ValueError(""),
        ValueError("\t \r\n"),
    ]
    empty = [
        f"{type(exc).__name__}{exc.args!r}"
        for exc in blank
        if not latched_load_error(exc).strip()
    ]
    assert empty == [], (
        f"these failures latch a blank `last_error`, which the Settings tab hands to "
        f"`notifyError` verbatim and shows as a toast occupying the screen and saying "
        f"nothing: {empty}"
    )


def test_a_padded_failure_message_latches_its_text_without_the_padding() -> None:
    latched = latched_load_error(RuntimeError("  The model file is missing.\n"))
    assert latched == "The model file is missing."


def test_every_provider_writes_the_latch_through_the_helper_or_clears_it() -> None:
    offenders = [
        f"{module}:{line} assigns `{source}`"
        for module, line, source, accepted in _latch_assignments()
        if not accepted
    ]
    assert offenders == [], (
        f"a load-error latch is `None` or a string that is not blank and nothing else (ADR 083); "
        f"route these through `{_LATCH_HELPER}`: {offenders}"
    )


def test_the_latch_walk_reaches_both_local_providers() -> None:
    reached = {module for module, _, _, _ in _latch_assignments()}
    assert _PROVIDER_MODULES <= reached, (
        f"the walk covers {sorted(reached)}, which misses "
        f"{sorted(_PROVIDER_MODULES - reached)}; a provider outside it is unchecked"
    )


def test_the_walk_accepts_every_shape_that_does_reach_the_helper() -> None:
    allowed = {
        "cleared": f"        self.{_LATCH_ATTRIBUTE} = None",
        "cleared under an annotation": f"        self.{_LATCH_ATTRIBUTE}: str | None = None",
        "the helper directly": f"        self.{_LATCH_ATTRIBUTE} = {_LATCH_HELPER}(e)",
        "a name bound once to the helper": (
            f"        msg = {_LATCH_HELPER}(e)\n        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a tuple target taking the helper": (
            f"        self.{_LATCH_ATTRIBUTE}, self.ready = {_LATCH_HELPER}(e), False"
        ),
    }
    classified = {
        label: _classified(_CLASS_BODY.format(snippet)) for label, snippet in allowed.items()
    }
    reported = sorted(
        label
        for label, written in classified.items()
        if not written or not all(accepted for _, accepted in written)
    )
    assert reported == [], (
        f"the walk reports a shape that does route through `{_LATCH_HELPER}`, so a provider "
        f"writing it correctly is told it did not: {reported} out of {classified}"
    )


def test_the_walk_rejects_every_shape_that_can_latch_a_blank_string() -> None:
    escapes = {
        "a tuple target": f"        self.{_LATCH_ATTRIBUTE}, self.ready = str(e), False",
        "an unresolvable unpacking": f"        self.{_LATCH_ATTRIBUTE}, self.ready = pair(e)",
        "an augmented assignment": f"        self.{_LATCH_ATTRIBUTE} += str(e)",
        "an annotated assignment": f"        self.{_LATCH_ATTRIBUTE}: str = str(e)",
        "a setattr call": f'        setattr(self, "{_LATCH_ATTRIBUTE}", str(e))',
        "a setattr naming its attribute at runtime": "        setattr(self, chosen(e), str(e))",
        "a for-loop target": (
            f"        for self.{_LATCH_ATTRIBUTE} in messages(e):\n            pass"
        ),
        "a comprehension target": (
            f"        kept = [m for self.{_LATCH_ATTRIBUTE} in messages(e)]"
        ),
        "a name a later assignment rebinds": (
            f"        msg = {_LATCH_HELPER}(e)\n"
            "        msg = str(e)\n"
            f"        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a name a later augmentation rebinds": (
            f"        msg = {_LATCH_HELPER}(e)\n"
            '        msg += "!"\n'
            f"        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a name a loop rebinds": (
            f"        msg = {_LATCH_HELPER}(e)\n"
            "        for msg in parts(e):\n            pass\n"
            f"        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a name a with block rebinds": (
            f"        msg = {_LATCH_HELPER}(e)\n"
            "        with opened(e) as msg:\n            pass\n"
            f"        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a parameter the body rebinds from the helper": (
            "        pass\n"
            f"    def store(self, e, msg):\n"
            f"        msg = {_LATCH_HELPER}(e)\n"
            f"        self.{_LATCH_ATTRIBUTE} = msg"
        ),
        "a name bound in a sibling method": (
            f"        self.{_LATCH_ATTRIBUTE} = msg\n"
            f"    def other(self, e):\n        msg = {_LATCH_HELPER}(e)"
        ),
    }
    missed = sorted(
        label
        for label, snippet in escapes.items()
        if not any(not accepted for _, accepted in _classified(_CLASS_BODY.format(snippet)))
    )
    assert missed == [], (
        f"these writes reach `{_LATCH_ATTRIBUTE}` without passing `{_LATCH_HELPER}` and the "
        f"walk waves them through, so a provider can latch a blank string unseen: {missed}"
    )


def test_the_walk_descends_into_a_subpackage_of_the_speech_layer(tmp_path) -> None:
    nested = tmp_path / "providers" / "vendor"
    nested.mkdir(parents=True)
    (nested / "later.py").write_text(
        f"class P:\n    def load(self, e):\n        self.{_LATCH_ATTRIBUTE} = str(e)\n",
        encoding="utf-8",
    )
    reached = [
        (module, accepted) for module, _, _, accepted in _latch_assignments(tmp_path)
    ]
    assert reached == [("providers/vendor/later.py", False)], (
        f"a provider added in a subpackage of the speech layer is walked like any other; "
        f"this walk reached {reached}"
    )
