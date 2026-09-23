"""Two anti-vacuity gates over `backend/tests`.

Rule 1 checks that a test can fail at all; rule 2 checks that a gate reporting
offenders out of a walk also pins that walk non-empty. Neither checks whether a
test's *name* describes what it asserts (ADR 079). Rule 2 resolves bindings by
heuristic and a shape it cannot resolve is out of scope, so it under-reports.
Only the Python suite is walked; the vitest files are not covered.
"""

import ast
import functools
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

_EMPTY_LITERALS = frozenset(
    {"[]", "()", "{}", "set()", "dict()", "list()", "tuple()", "frozenset()"}
)

_RAISES_CONTEXTS = ("pytest.raises", "pytest.warns", "pytest.deprecated_call")

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

_NO_BINDINGS: dict[str, list[ast.AST]] = {}


def _source_files() -> tuple[Path, ...]:
    """Every Python module of the backend suite, `__pycache__` excluded."""
    return tuple(
        path
        for path in sorted(_TESTS_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


@functools.cache
def _module_trees() -> tuple[tuple[str, ast.Module], ...]:
    """Every suite module as (module, parsed tree)."""
    return tuple(
        (path.relative_to(_TESTS_DIR).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in _source_files()
    )


@functools.cache
def _test_functions() -> tuple[tuple[str, str, ast.AST], ...]:
    """Every `test_`-prefixed function in the suite as (module, name, node)."""
    found: list[tuple[str, str, ast.AST]] = []
    for module, tree in _module_trees():
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTIONS) and node.name.startswith("test_"):
                found.append((module, node.name, node))
    return tuple(found)


def _callee_name(call: ast.Call) -> str | None:
    """The bare name a call resolves to, dropping any dotted prefix."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_constant_truthy(node: ast.AST) -> bool:
    """True when the expression is a literal that can never be falsy."""
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _is_trivially_true(node: ast.AST) -> bool:
    """True when no call inside the expression can change its truth."""
    if _is_constant_truthy(node):
        return True
    if not isinstance(node, ast.BoolOp):
        return False
    if isinstance(node.op, ast.Or):
        return any(_is_trivially_true(value) for value in node.values)
    return all(_is_trivially_true(value) for value in node.values)


def _contains_an_assertion(node: ast.AST) -> bool:
    """Whether a test body holds any of the four assertion shapes this suite uses.

    A nested `raise AssertionError` counts: it is how a monkeypatched stub says
    the call it replaced must never happen, and it fails the test when it does.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert) and not _is_trivially_true(sub.test):
            return True
        if isinstance(sub, ast.Call):
            name = _callee_name(sub)
            if name is not None and name.lstrip("_").startswith("assert"):
                return True
        if isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                text = ast.unparse(item.context_expr)
                if any(text.startswith(context) for context in _RAISES_CONTEXTS):
                    return True
        if isinstance(sub, ast.Raise) and sub.exc is not None:
            raised = sub.exc.func if isinstance(sub.exc, ast.Call) else sub.exc
            if ast.unparse(raised).split(".")[-1] == "AssertionError":
                return True
    return False


def _module_helpers(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level functions that are not themselves tests."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, _FUNCTIONS) and not node.name.startswith("test_")
    }


def _non_empty_module_constants(tree: ast.Module) -> frozenset[str]:
    """Module-level names bound to a literal that is not empty."""
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        literal = _literal_or_none(value)
        if literal:
            found.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(found)


def _literal_or_none(value: ast.AST) -> object:
    """The literal a node evaluates to, seeing through one builtin container call."""
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(value, ast.Call) and _callee_name(value) in {
        "set",
        "frozenset",
        "list",
        "tuple",
        "dict",
    }:
        if value.args:
            return _literal_or_none(value.args[0])
    return None


def _pins_its_own_result(function: ast.AST) -> bool:
    """Whether a walk asserts its own result non-empty on every path out of it.

    `assert found` above `return tuple(found)` fires on every call, so it pins the
    walk harder than a separate test can. An assert behind a branch or below any
    return does not fire on every call, and one such helper would silence the rule
    for every consumer in its module, so neither counts.
    """
    returns = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    if not returns:
        return False
    for index, statement in enumerate(function.body):
        if not (isinstance(statement, ast.Assert) and isinstance(statement.test, ast.Name)):
            continue
        if any(
            isinstance(node, ast.Return)
            for earlier in function.body[:index]
            for node in ast.walk(earlier)
        ):
            return False
        asserted = statement.test.id
        if all(
            any(
                isinstance(sub, ast.Name) and sub.id == asserted
                for sub in ast.walk(node.value)
            )
            for node in returns
        ):
            return True
    return False


def _helper_closure(helpers: dict[str, ast.AST]) -> dict[str, frozenset[str]]:
    """Each helper mapped to itself plus every helper it transitively calls."""
    direct = {
        name: {
            called
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and (called := _callee_name(sub)) in helpers
        }
        for name, node in helpers.items()
    }
    closure: dict[str, frozenset[str]] = {}
    for name in helpers:
        seen: set[str] = set()
        pending = [name]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(direct[current])
        closure[name] = frozenset(seen)
    return closure


def _local_bindings(function: ast.AST) -> dict[str, list[ast.AST]]:
    """Every expression each local name is built from, assignment and mutation alike."""
    bindings: dict[str, list[ast.AST]] = {}
    for sub in ast.walk(function):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(sub.value)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if sub.value is not None:
                bindings.setdefault(sub.target.id, []).append(sub.value)
        elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
            bindings.setdefault(sub.target.id, []).append(sub.value)
        elif isinstance(sub, (ast.For, ast.AsyncFor)) and isinstance(sub.target, ast.Name):
            bindings.setdefault(sub.target.id, []).append(sub.iter)
        elif isinstance(sub, ast.comprehension) and isinstance(sub.target, ast.Name):
            bindings.setdefault(sub.target.id, []).append(sub.iter)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"append", "extend", "add", "update"}
            and isinstance(sub.func.value, ast.Name)
            and sub.args
        ):
            bindings.setdefault(sub.func.value.id, []).append(sub.args[0])
    return bindings


def _nodes_under_raises(function: ast.AST) -> frozenset[int]:
    """Every node id inside a `pytest.raises` block, which is out of scope."""
    guarded: set[int] = set()
    for sub in ast.walk(function):
        if not isinstance(sub, (ast.With, ast.AsyncWith)):
            continue
        texts = [ast.unparse(item.context_expr) for item in sub.items]
        if not any(text.startswith("pytest.raises") for text in texts):
            continue
        for statement in sub.body:
            guarded.update(id(child) for child in ast.walk(statement))
    return frozenset(guarded)


def _helpers_reached(
    expr: ast.AST,
    bindings: dict[str, list[ast.AST]],
    closure: dict[str, frozenset[str]],
    guarded: frozenset[int],
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """The same-module helpers an expression consumes, following bindings three deep."""
    reached: set[str] = set()
    for sub in ast.walk(expr):
        if id(sub) in guarded:
            continue
        if isinstance(sub, ast.Call) and (called := _callee_name(sub)) in closure:
            reached |= closure[called]
        if isinstance(sub, ast.Name) and sub.id in bindings and depth < 3 and sub.id not in seen:
            for value in bindings[sub.id]:
                reached |= _helpers_reached(
                    value, bindings, closure, guarded, depth + 1, seen | {sub.id}
                )
    return frozenset(reached)


def _emptiness_target(test: ast.AST) -> ast.AST | None:
    """The collection an emptiness-shaped assertion claims is empty, if any."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return test.operand
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    if not isinstance(test.ops[0], ast.Eq):
        return None
    for side, other in ((test.left, test.comparators[0]), (test.comparators[0], test.left)):
        if ast.unparse(other).replace(" ", "") in _EMPTY_LITERALS:
            return side
        if (
            isinstance(side, ast.Call)
            and _callee_name(side) == "len"
            and side.args
            and isinstance(other, ast.Constant)
            and other.value == 0
        ):
            return side.args[0]
    return None


def _resolves_to_non_empty_constant(
    expr: ast.AST,
    bindings: dict[str, list[ast.AST]],
    constants: frozenset[str],
    depth: int = 0,
) -> bool:
    """Whether an expression bottoms out in a module-level literal that is not empty."""
    if isinstance(expr, ast.Name):
        if expr.id in constants:
            return True
        if depth >= 3:
            return False
        return any(
            _resolves_to_non_empty_constant(value, bindings, constants, depth + 1)
            for value in bindings.get(expr.id, [])
        )
    if isinstance(expr, ast.Call) and expr.args:
        return _resolves_to_non_empty_constant(expr.args[0], bindings, constants, depth)
    return bool(_literal_or_none(expr))


def _is_self_pinning(
    target: ast.AST,
    bindings: dict[str, list[ast.AST]],
    constants: frozenset[str],
    depth: int = 0,
) -> bool:
    """`A - B` with A a non-empty constant fails on an empty walk, so it pins itself."""
    if isinstance(target, ast.Name):
        if depth >= 3:
            return False
        return any(
            _is_self_pinning(value, bindings, constants, depth + 1)
            for value in bindings.get(target.id, [])
        )
    if isinstance(target, ast.BinOp) and isinstance(target.op, ast.Sub):
        return _resolves_to_non_empty_constant(target.left, bindings, constants)
    if isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute):
        if target.func.attr == "difference":
            return _resolves_to_non_empty_constant(target.func.value, bindings, constants)
    return False


def _unpinned_walks(tree: ast.Module) -> list[tuple[int, str, tuple[str, ...]]]:
    """Emptiness assertions consuming a walk that nothing in the module pins non-empty.

    Returns (line, test name, walks) per offender. Every assertion sorts into
    exactly one of pin, offender and out of scope, and an emptiness-shaped one
    never becomes a pin however its target is written (ADR 079).
    """
    helpers = _module_helpers(tree)
    if not helpers:
        return []
    closure = _helper_closure(helpers)
    constants = _non_empty_module_constants(tree)

    pinned = {name for name, node in helpers.items() if _pins_its_own_result(node)}
    candidates: list[tuple[ast.Assert, str, frozenset[str]]] = []
    for function in ast.walk(tree):
        if not (isinstance(function, _FUNCTIONS) and function.name.startswith("test_")):
            continue
        bindings = _local_bindings(function)
        guarded = _nodes_under_raises(function)
        for sub in ast.walk(function):
            if not isinstance(sub, ast.Assert):
                continue
            target = _emptiness_target(sub.test)
            if target is None:
                if not _is_trivially_true(sub.test):
                    pinned |= _helpers_reached(sub.test, bindings, closure, guarded)
                continue
            if _is_self_pinning(target, bindings, constants):
                continue
            if isinstance(target, ast.Name) and target.id in bindings:
                reached = _helpers_reached(target, bindings, closure, guarded)
            else:
                reached = _helpers_reached(target, _NO_BINDINGS, closure, guarded)
            if reached:
                candidates.append((sub, function.name, reached))

    return [
        (node.lineno, name, tuple(sorted(reached)))
        for node, name, reached in candidates
        if not (reached & pinned)
    ]


def test_every_test_contains_an_assertion_that_can_fail():
    silent = [
        f"{module}:{name}"
        for module, name, node in _test_functions()
        if not _contains_an_assertion(node)
    ]
    assert not silent, (
        "a test that cannot fail is not coverage (ADR 079) -- give each of these an "
        f"assertion on an observable outcome rather than a marker: {silent}"
    )


def test_no_gate_passes_on_a_walk_that_found_nothing():
    unpinned = [
        f"{module}:{line} {name} consumes {', '.join(w + '()' for w in walks)}"
        for module, tree in _module_trees()
        for line, name, walks in _unpinned_walks(tree)
    ]
    assert not unpinned, (
        "an emptiness assertion over a walk passes when the walk goes blind (ADR 079); "
        "add a membership pin to each module, on the model of test_naming_rules.py::"
        f"test_the_walk_finds_the_declarations_it_is_meant_to_check: {unpinned}"
    )


def test_the_assertion_recogniser_separates_a_silent_test_from_a_failing_one():
    """A test with no assertion that can fail is reported; one with any of the four is not."""
    silent = (
        "def test_a():\n    pass\n",
        "def test_b():\n    assert True\n",
        "def test_c():\n    result = compute()\n    print(result)\n",
        "def test_d():\n    assert 1\n",
        "def test_e():\n    assert True or compute()\n",
        "def test_f():\n    assert True and 1\n",
    )
    asserting = (
        "def test_a():\n    assert result == 3\n",
        "def test_b():\n    with pytest.raises(ValueError):\n        boom()\n",
        "def test_c():\n    with pytest.warns(UserWarning):\n        warn()\n",
        "def test_d():\n    self.assertEqual(a, b)\n",
        "def test_e():\n    _assert_shape(a)\n",
        "def test_f():\n"
        "    def _boom():\n"
        "        raise AssertionError('must not be called')\n"
        "    monkeypatch.setattr(mod, 'f', _boom)\n"
        "    mod.run()\n",
    )
    assert not _contains_an_assertion(ast.parse(silent[0]).body[0]), (
        "a test whose whole body is `pass` cannot fail and must be reported"
    )
    assert _contains_an_assertion(ast.parse(asserting[-1]).body[0]), (
        "a monkeypatched stub raising AssertionError is an assertion: it fails the "
        "test when the call it replaced happens"
    )
    missed = [s for s in silent if _contains_an_assertion(ast.parse(s).body[0])]
    flagged = [s for s in asserting if not _contains_an_assertion(ast.parse(s).body[0])]
    assert not missed, f"these tests cannot fail and the rule walks past them: {missed}"
    assert not flagged, f"these assert an outcome and must not be reported: {flagged}"


def test_the_walk_rule_separates_an_unpinned_gate_from_a_self_pinning_one():
    """The compliant shapes are `<non-empty module constant> - <walk>`, a sibling
    membership pin, a walk outside the module, and a helper asserting its own result."""
    unpinned = (
        "def _walk():\n    return []\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
        "def _walk():\n    return []\n\n\n"
        "def test_a():\n    assert True or _walk()\n\n\n"
        "def test_b():\n    offenders = list(_walk())\n    assert not offenders\n",
        "def _walk():\n    return []\n\n\n"
        "def test_a():\n    offenders = list(_walk())\n    assert offenders == []\n",
        "def _walk():\n    return []\n\n\n"
        "def test_a():\n    offenders = list(_walk())\n    assert len(offenders) == 0\n",
        "def _walk():\n"
        "    found = []\n"
        "    if _sometimes():\n"
        "        assert found\n"
        "    return tuple(found)\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
        "def _walk():\n"
        "    found = []\n"
        "    return tuple(found)\n"
        "    assert found\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
        "def _walk():\n"
        "    found = []\n"
        "    if _sometimes():\n"
        "        return tuple(found)\n"
        "    assert found\n"
        "    return tuple(found)\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
        "def _walk():\n"
        "    found = []\n"
        "    unrelated = [1]\n"
        "    assert unrelated\n"
        "    return tuple(found)\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
    )
    pinned = (
        "EXPECTED = {'a', 'b'}\n\n\n"
        "def _walk():\n    return set()\n\n\n"
        "def test_a():\n"
        "    discovered = _walk()\n"
        "    expected = set(EXPECTED)\n"
        "    stale = expected - discovered\n"
        "    assert not stale\n",
        "def _walk():\n    return []\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n\n\n"
        "def test_b():\n    assert {'a', 'b'} <= set(_walk())\n",
        "def test_a():\n    offenders = [x for x in walk() if x]\n    assert not offenders\n",
        "def _walk():\n"
        "    found = []\n"
        "    assert found\n"
        "    return tuple(found)\n\n\n"
        "def test_a():\n    offenders = [x for x in _walk() if x]\n    assert not offenders\n",
    )
    inlined = (
        "def _walk():\n"
        "    return []\n\n\n"
        "def test_a():\n"
        "    offenders = [x for x in _walk() if x]\n"
        "    assert not offenders\n\n\n"
        "def test_b():\n"
        "    assert not _walk()\n"
    )
    assert not _unpinned_walks(ast.parse(pinned[0])), (
        "`<non-empty module constant> - <walk>` fails when the walk goes blind, which is "
        "the property the rule asks for"
    )
    reported = [(name, walks) for _, name, walks in _unpinned_walks(ast.parse(unpinned[0]))]
    assert reported == [("test_a", ("_walk",))], (
        f"the rule must name the test and the walk it left unpinned, not just fail: {reported}"
    )
    both = _unpinned_walks(ast.parse(inlined))
    assert both == [(7, "test_a", ("_walk",)), (11, "test_b", ("_walk",))], (
        "deleting the intermediate variable is the cheap wrong fix, and it must add a "
        "second offender rather than pin the module: an emptiness-shaped assertion is "
        f"never a pin, whatever its target looks like (ADR 079). Reported: {both}"
    )
    missed = [s for s in unpinned if not _unpinned_walks(ast.parse(s))]
    flagged = [s for s in pinned if _unpinned_walks(ast.parse(s))]
    assert not missed, f"these gates pass on a blind walk and the rule misses them: {missed}"
    assert not flagged, (
        f"these walks are pinned or out of scope and must not be reported: {flagged}"
    )


def test_the_walk_reaches_every_module_family_it_is_meant_to_check():
    """Every rule above passes vacuously on an empty walk, so pin that it is not."""
    modules = {module for module, _, _ in _test_functions()}
    swept = {
        "test_audio.py",
        "test_import_layers.py",
        "test_vector_store.py",
        "test_docstring_budget.py",
        "test_cross_language_contracts.py",
    }
    assert swept <= modules
