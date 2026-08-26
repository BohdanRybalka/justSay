"""Every module-level loop-bound asyncio primitive in `app/` must be reset
between tests.

`Lock`, `Event`, `Condition`, the semaphores and the queues all inherit
`_LoopBoundMixin`, which stores the first event loop reaching `_get_loop()`
and raises `RuntimeError: ... is bound to a different event loop` for every
loop after that. The binding happens on a genuinely contended `acquire()`
only -- the uncontended fast path returns first -- so a module-level singleton
can sit unbound through hundreds of tests and then trap the first two that
actually contend it.

That is how JS-110 reached CI: `_prewarm_lock` was reset by a fixture local to
`test_local_setup.py`, while the readiness-barrier test in `test_pipeline.py`
contended the same singleton with no reset of its own. It passed only because
`test_local_setup.py` sorts first.

An AST scan rather than a behavioural pin, for the same reason
`test_sys_modules_hygiene.py` uses one: the damage needs two specific tests in
one specific order, so a behavioural test would be flaky by construction. This
asks the structural question instead -- is every loop-bound global registered
for reset -- and that has one answer regardless of ordering.

**A scan is only worth having if it cannot miss quietly.** The first draft
matched one shape, `name = asyncio.Lock()` at the top level of a module, and
the PR #79 review found four ways past it: an annotated assignment
(`name: T = asyncio.Lock()`), an aliased import (`from asyncio import Lock`),
an assignment nested in a module-level `if`/`try`, and tuple unpacking. All
four are covered here, and a third assertion closes the shape that cannot be
covered at all: a primitive constructed at module level without being bound to
a plain name -- inside a dict, a list, an attribute -- is unreachable by a
reset that works by attribute name, so it fails rather than passing silently.
"""

import ast
from pathlib import Path

from tests.conftest import EVENT_LOOP_BOUND_LOCKS

_APP_DIR = Path(__file__).parent.parent / "app"

_LOOP_BOUND_TYPES = frozenset(
    {
        "BoundedSemaphore",
        "Condition",
        "Event",
        "LifoQueue",
        "Lock",
        "PriorityQueue",
        "Queue",
        "Semaphore",
    }
)

_ASYNCIO_MODULES = frozenset({"asyncio", "asyncio.locks", "asyncio.queues"})

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _module_path(path: Path) -> str:
    relative = path.relative_to(_APP_DIR.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _asyncio_names(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """Local names bound to the asyncio module, and to a loop-bound type."""
    modules: set[str] = set()
    types: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _ASYNCIO_MODULES:
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module in _ASYNCIO_MODULES:
            for alias in node.names:
                if alias.name in _LOOP_BOUND_TYPES:
                    types[alias.asname or alias.name] = alias.name
    return modules, types


def _constructed_type(node: ast.AST, modules: set[str], types: dict[str, str]) -> str | None:
    """The loop-bound type this node constructs, if it constructs one."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return types.get(func.id)
    if isinstance(func, ast.Attribute) and func.attr in _LOOP_BOUND_TYPES:
        owner = func.value
        if isinstance(owner, ast.Name) and owner.id in modules:
            return func.attr
        if (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id in modules
        ):
            return func.attr
    return None


def _module_scope_statements(body):
    """Statements reached without entering a function or class body."""
    for node in body:
        if isinstance(node, _NESTED_SCOPES):
            continue
        yield node
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(node, field, None)
            if isinstance(nested, list):
                yield from _module_scope_statements(nested)
        for handler in getattr(node, "handlers", []):
            yield from _module_scope_statements(handler.body)


def _nodes_outside_nested_scopes(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NESTED_SCOPES):
            continue
        yield child
        yield from _nodes_outside_nested_scopes(child)


def _bindings(statement: ast.stmt, modules: set[str], types: dict[str, str]):
    """`(name, type, constructing call)` for what this statement binds."""
    if isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
        value = statement.value
    elif isinstance(statement, ast.Assign):
        targets = statement.targets
        value = statement.value
    else:
        return

    if value is None:
        return

    for target in targets:
        unpacks = isinstance(target, ast.Tuple | ast.List) and isinstance(
            value, ast.Tuple | ast.List
        )
        pairs = zip(target.elts, value.elts) if unpacks else iter([(target, value)])
        for name_node, value_node in pairs:
            type_name = _constructed_type(value_node, modules, types)
            if type_name and isinstance(name_node, ast.Name):
                yield name_node.id, type_name, value_node


def _module_level_primitives(tree: ast.Module) -> tuple[dict[str, str], list[int]]:
    """Names bound to a loop-bound primitive, plus the lines of any module-level
    construction that no such name accounts for."""
    modules, types = _asyncio_names(tree)
    bound: dict[str, str] = {}
    accounted: set[int] = set()

    for statement in _module_scope_statements(tree.body):
        for name, type_name, call in _bindings(statement, modules, types):
            bound[name] = type_name
            accounted.add(id(call))

    unresettable = sorted(
        node.lineno
        for node in _nodes_outside_nested_scopes(tree)
        if _constructed_type(node, modules, types) and id(node) not in accounted
    )
    return bound, unresettable


def _scan_app() -> tuple[dict[tuple[str, str], str], dict[str, list[int]]]:
    found: dict[tuple[str, str], str] = {}
    unresettable: dict[str, list[int]] = {}
    for path in sorted(_APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = _module_path(path)
        bound, loose = _module_level_primitives(tree)
        for name, type_name in bound.items():
            found[module, name] = type_name
        if loose:
            unresettable[module] = loose
    return found, unresettable


def test_every_module_level_asyncio_lock_is_reset_between_tests():
    found, unresettable = _scan_app()
    registered = set(EVENT_LOOP_BOUND_LOCKS)

    missing = sorted(set(found) - registered)
    stale = sorted(registered - set(found))

    assert not missing, (
        "these module-level loop-bound asyncio primitives are never reset between "
        f"tests: {[(module, name, found[module, name]) for module, name in missing]}. "
        "The first test to contend one binds it to that test's event loop for the "
        "rest of the session, and the next test to contend it raises 'bound to a "
        "different event loop'. Add each to conftest.EVENT_LOOP_BOUND_LOCKS -- which "
        "rebinds an asyncio.Lock(), so anything else needs the fixture extended too."
    )
    assert not stale, (
        "conftest.EVENT_LOOP_BOUND_LOCKS resets locks that no longer exist in "
        f"app/: {stale}. Remove them."
    )
    assert not unresettable, (
        "these modules construct a loop-bound asyncio primitive at module level "
        f"without binding it to a plain name (module: lines): {unresettable}. "
        "conftest._reset_event_loop_bound_locks resets by attribute name, so an "
        "object held in a dict, a list or an attribute can never be reset at all. "
        "Bind it to a module-level name and register it, or construct it inside "
        "the function that uses it."
    )
