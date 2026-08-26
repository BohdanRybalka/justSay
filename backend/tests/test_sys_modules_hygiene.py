"""No test may remove a module from `sys.modules`.

Deleting and re-importing binds a NEW module object. Restoring only the
`sys.modules` entry does not undo it: the parent package's attribute still
points at the replacement, so `monkeypatch.setattr("pkg.mod.name", ...)`
patches one object while a `from pkg.mod import name` inside the code under
test reads the other. Partial restoration is therefore worse than none.

That is not hypothetical. Two import-hygiene tests did it, and fixing them
the obvious way -- `monkeypatch.delitem`, which restores the `sys.modules`
entry alone -- made four local-STT tests resolve the real, unpatched provider
routing and fail on this AMD dev box, while the suite slowed from 75 s to
319 s. Both now check the import in a subprocess instead
(`conftest.assert_module_binds_no_third_party`), which touches no global state
and asks a stricter question anyway.

An AST scan rather than a behavioural pin on purpose: the damage is
order-dependent and invisible in any single ordering, so a behavioural test
would be flaky by construction.
"""

import ast
from pathlib import Path

_TESTS_DIR = Path(__file__).parent


def _sys_modules_removals(tree: ast.AST) -> list[int]:
    """Line numbers of `del sys.modules[...]` and `sys.modules.pop(...)`."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "modules"
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "sys"
                ):
                    lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "pop"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "modules"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                lines.append(node.lineno)
    return lines


def test_no_test_removes_a_module_from_sys_modules():
    offenders: dict[str, list[int]] = {}
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = _sys_modules_removals(tree)
        if lines:
            offenders[path.name] = lines

    assert not offenders, (
        "these tests remove a module from sys.modules (file: line numbers): "
        f"{offenders}. Restoring the entry is not enough -- the parent package "
        "attribute keeps pointing at the replacement. Import in a subprocess "
        "instead: conftest.assert_module_binds_no_third_party."
    )
