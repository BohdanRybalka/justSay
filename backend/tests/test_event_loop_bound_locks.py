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


def _module_path(path: Path) -> str:
    relative = path.relative_to(_APP_DIR.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _loop_bound_globals(tree: ast.Module) -> list[tuple[str, str]]:
    """`(name, asyncio type)` for every module-level loop-bound assignment."""
    found: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if (
            not isinstance(func, ast.Attribute)
            or func.attr not in _LOOP_BOUND_TYPES
            or not isinstance(func.value, ast.Name)
            or func.value.id != "asyncio"
        ):
            continue
        found.extend(
            (target.id, func.attr) for target in node.targets if isinstance(target, ast.Name)
        )
    return found


def _declared_in_app() -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for path in sorted(_APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = _module_path(path)
        for name, type_name in _loop_bound_globals(tree):
            found[(module, name)] = type_name
    return found


def test_every_module_level_asyncio_lock_is_reset_between_tests():
    found = _declared_in_app()
    registered = set(EVENT_LOOP_BOUND_LOCKS)

    unregistered = sorted(set(found) - registered)
    stale = sorted(registered - set(found))

    assert not unregistered, (
        "these module-level loop-bound asyncio primitives are never reset between "
        f"tests: {[(module, name, found[module, name]) for module, name in unregistered]}. "
        "The first test to contend one binds it to that test's event loop for the "
        "rest of the session, and the next test to contend it raises 'bound to a "
        "different event loop'. Add each to conftest.EVENT_LOOP_BOUND_LOCKS -- which "
        "rebinds an asyncio.Lock(), so anything else needs the fixture extended too."
    )
    assert not stale, (
        "conftest.EVENT_LOOP_BOUND_LOCKS resets locks that no longer exist in "
        f"app/: {stale}. Remove them."
    )
