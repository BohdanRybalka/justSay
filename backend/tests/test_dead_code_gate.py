"""The vulture gate must report an uncalled symbol, not merely exit 0.

`min_confidence = 100` reports unreachable code and unused arguments only:
vulture scores an unused function, class, method or attribute at 60, so the
gate ADR 017 made PR-blocking was green while blind to the class of dead code
it exists to catch (ADR 051). These tests run the real
`backend/pyproject.toml` against synthetic packages, so setting
`min_confidence` back to 100 -- or deleting `ignore_decorators` -- turns them
red instead of turning the gate quietly blind again.

`tomli` rather than stdlib `tomllib` below Python 3.11: the local
interpreter is 3.10, and `tomli` is a hard dependency of vulture 2.16
itself, so it is installed wherever this gate can run at all.

The subprocess takes the config by `--config <the real pyproject.toml>` and a
positional path that overrides the config's own `paths`, so what is exercised
is the shipped table rather than a copy of its values.
"""

import ast
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PYPROJECT = _BACKEND_DIR / "pyproject.toml"
_APP_DIR = _BACKEND_DIR / "app"

_VULTURE_FOUND_DEAD_CODE = 3
_VULTURE_TIMEOUT_SECONDS = 60


def _vulture_table() -> dict:
    """The shipped `[tool.vulture]` table, read from the real pyproject.toml."""
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["vulture"]


def _run_vulture(target: Path) -> subprocess.CompletedProcess[str]:
    """Run vulture over `target` under the shipped config.

    The timeout is the same guard `conftest._run_import_probe` carries: a
    child that wedges rather than finishing must fail the test that asked for
    it, not hang the suite.
    """
    try:
        return subprocess.run(
            [sys.executable, "-m", "vulture", str(target), "--config", str(_PYPROJECT)],
            capture_output=True,
            text=True,
            timeout=_VULTURE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(
            f"vulture did not finish within {_VULTURE_TIMEOUT_SECONDS}s over {target} -- "
            "it wedged rather than reporting, which would otherwise hang the whole suite"
        ) from expired


def _write_package(root: Path, modules: dict[str, str]) -> Path:
    """Write a synthetic importable package and return its directory."""
    package = root / "synthetic_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, source in modules.items():
        (package / f"{name}.py").write_text(source, encoding="utf-8")
    return package


def test_gate_reports_a_module_level_function_that_nothing_calls(tmp_path):
    """The whole point of the retune: an uncalled function is named, exit 3."""
    package = _write_package(
        tmp_path,
        {
            "used": "def called_by_the_other_module():\n    return 1\n",
            "caller": (
                "from synthetic_pkg.used import called_by_the_other_module\n\n\n"
                "def uncalled_by_anything():\n    return 2\n\n\n"
                "def entry():\n    return called_by_the_other_module()\n"
            ),
        },
    )

    result = _run_vulture(package)

    assert result.returncode == _VULTURE_FOUND_DEAD_CODE, (
        "vulture did not report the uncalled function -- the gate is blind again. "
        f"exit {result.returncode}:\n{result.stdout}{result.stderr}"
    )
    assert "uncalled_by_anything" in result.stdout, (
        f"expected 'uncalled_by_anything' in vulture's output:\n{result.stdout}{result.stderr}"
    )
    assert "called_by_the_other_module" not in result.stdout, (
        f"vulture reported a function that has a caller:\n{result.stdout}{result.stderr}"
    )


def test_router_decorated_handlers_are_exempt(tmp_path):
    """`ignore_decorators = ["@router.*"]` is what makes confidence 60 affordable.

    It removes 30 of the 70 findings by rule rather than by name, so a new
    endpoint costs the allowlist nothing. Without it every FastAPI handler in
    the backend would be reported as uncalled.
    """
    package = _write_package(
        tmp_path,
        {
            "routes": (
                "class _Router:\n"
                "    def get(self, path):\n"
                "        return lambda fn: fn\n\n\n"
                "router = _Router()\n\n\n"
                '@router.get("/x")\n'
                "def handler_nothing_calls():\n"
                "    return {}\n"
            ),
        },
    )

    result = _run_vulture(package)

    assert "handler_nothing_calls" not in result.stdout, (
        "a @router-decorated handler was reported as dead -- the ignore_decorators "
        f"exemption is gone, and every endpoint now costs an allowlist entry:\n{result.stdout}"
    )


def test_min_confidence_is_low_enough_to_see_an_unused_function():
    """60 is the tier at which functions, classes, methods and attributes are scored.

    Above it vulture reports unused imports (90, already ruff's `F401`) and
    unreachable code only.
    """
    assert _vulture_table()["min_confidence"] <= 60, (
        "vulture scores an unused function, class, method or attribute at 60. "
        "Any higher min_confidence makes the gate blind to all four -- see ADR 051."
    )


def _bound_names(path: Path) -> set[str]:
    """Every name `path` actually binds, collected from its AST.

    A textual scan cannot tell a field declaration from the same word inside
    a docstring, so it reports an entry as live when only prose mentions it.
    The node kinds below are what the allowlist's own four groups need:
    definitions and imports, `Store`-context names (assignments, annotated
    pydantic fields, `for`/`with`/walrus/comprehension targets), `Store`
    attributes (`conn.row_factory = ...`), and arguments (the sounddevice
    callback contract's `time_info`/`status`/`frame_count`).

    A name can also be declared as data rather than as syntax -- the Win32
    job-object struct declares `LimitFlags` as a string inside
    `ctypes.Structure._fields_` (`app/stt/local_whisper_cpp.py:124`), and
    `__slots__` and pydantic aliases do the same. A string constant that is
    exactly an identifier therefore counts as a binding, except where it is a
    docstring: that exception is what keeps this check structural, since the
    prose it must not match is never a bare identifier.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise AssertionError(
            f"{path} could not be parsed, so its bound names are invisible to this "
            f"check and any allowlist entry living there would read as stale: {error}"
        ) from error

    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Store):
            names.add(node.id if isinstance(node, ast.Name) else node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isidentifier()
            and id(node) not in docstrings
        ):
            names.add(node.value)
    return names


def test_no_ignore_names_entry_has_outlived_its_symbol():
    """An allowlist entry whose symbol is gone silently widens the gate.

    `ignore_names` suppresses a name everywhere under `app`, so a stale entry
    is a permanent blind spot for any future symbol that happens to reuse the
    name. Nothing else notices one, because removing the symbol is exactly
    what leaves the gate green.

    The check is structural rather than textual: an entry counts as live only
    while some file under `app` still *binds* that name. Matching raw text
    instead would let a docstring keep an entry alive after its symbol was
    deleted, which is the one failure this test exists to catch.
    """
    bound = set()
    for path in sorted(_APP_DIR.rglob("*.py")):
        bound |= _bound_names(path)

    stale = [name for name in _vulture_table()["ignore_names"] if name not in bound]

    assert not stale, (
        f"these [tool.vulture] ignore_names entries are bound by no file under "
        f"{_APP_DIR.name}/: {stale}. The symbol each was added for is gone, so the "
        "entry now only suppresses whatever future symbol reuses the name. "
        "Delete the entry."
    )
