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

import re
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
_MIRRORED_CONFIG_KEYS = {"paths", "min_confidence", "ignore_decorators", "exclude"}

_REPORTED_NAME = re.compile(r"unused \w+ '([^']+)'")


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

    The package carries an undecorated dead function beside the decorated one
    so absence is never the whole assertion. A vulture that fails to start --
    a missing module, a config key a future version rejects, a renamed flag --
    prints nothing and exits non-zero, and a test whose only claim is "this
    name is absent from stdout" passes on that while proving nothing.
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
            "plain": "def undecorated_nothing_calls():\n    return {}\n",
        },
    )

    result = _run_vulture(package)

    assert result.returncode == _VULTURE_FOUND_DEAD_CODE, (
        f"vulture did not report the undecorated dead function (exit {result.returncode}), "
        "so this test proves nothing about the decorator exemption:\n"
        f"{result.stdout}{result.stderr}"
    )
    assert "undecorated_nothing_calls" in result.stdout, (
        "vulture ran but reported no dead function at all, so the absence asserted below "
        f"is not evidence of anything:\n{result.stdout}{result.stderr}"
    )
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


def _names_vulture_reports_without_the_allowlist(cwd: Path) -> set[str]:
    """Every name vulture reports over the shipped paths with `ignore_names` emptied.

    Every key of the shipped table except `ignore_names` is mirrored onto the
    command line, so the run matches the gate in every respect but the
    allowlist. `--config` is deliberately not passed and `cwd` is a directory
    holding no `pyproject.toml`, because vulture auto-discovers one from the
    working directory and would re-apply the very `ignore_names` this run must
    not see.

    Mirroring by hand is only sound while the mirror is complete, so a key this
    function does not know fails the test rather than being ignored: a `paths`
    narrowed to one package, or an `exclude` added, would otherwise leave the
    gate scanning less than this run does and the set equality green over a
    gate that had gone blind.
    """
    table = _vulture_table()
    unmirrored = set(table) - _MIRRORED_CONFIG_KEYS - {"ignore_names"}
    assert not unmirrored, (
        f"[tool.vulture] carries keys this check does not mirror: {sorted(unmirrored)}. "
        "The run below would then differ from the gate itself, and its set equality would "
        "say nothing about the gate. Mirror the key here, or state why it cannot change "
        "what the gate scans."
    )

    command = [sys.executable, "-m", "vulture"]
    command += [str((_BACKEND_DIR / path).resolve()) for path in table["paths"]]
    command += ["--min-confidence", str(table["min_confidence"])]
    decorators = table.get("ignore_decorators", [])
    if decorators:
        command += ["--ignore-decorators", ",".join(decorators)]
    excluded = table.get("exclude", [])
    if excluded:
        command += ["--exclude", ",".join(excluded)]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=_VULTURE_TIMEOUT_SECONDS, cwd=str(cwd)
        )
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(
            f"vulture did not finish within {_VULTURE_TIMEOUT_SECONDS}s over {_APP_DIR} -- "
            "it wedged rather than reporting, which would otherwise hang the whole suite"
        ) from expired

    assert result.returncode in (0, _VULTURE_FOUND_DEAD_CODE), (
        f"vulture failed to run (exit {result.returncode}):\n{result.stdout}{result.stderr}"
    )
    return set(_REPORTED_NAME.findall(result.stdout))


def test_ignore_names_lists_exactly_what_the_gate_would_otherwise_report(tmp_path):
    """The allowlist must equal the findings it exists to suppress.

    vulture already answers this exactly: run the shipped table with
    `ignore_names` emptied and it prints precisely the names the allowlist
    silences. Set equality against that run catches both failure directions at
    once -- an entry whose symbol is gone (a permanent blind spot for whatever
    future symbol reuses the name) and a name the gate reports that nobody
    listed (real dead code, or an allowlist entry someone forgot to add).

    This replaces an AST collector that approximated the same question and got
    it wrong in three ways: counting any parameter anywhere made `status` and
    `word` unfalsifiable, an identifier-shaped string constant counted as a
    binding, and an `fnmatch` glob entry -- a form vulture supports -- would
    have read as stale.
    """
    reported = _names_vulture_reports_without_the_allowlist(tmp_path)
    allowlisted = set(_vulture_table()["ignore_names"])

    stale = sorted(allowlisted - reported)
    unlisted = sorted(reported - allowlisted)

    assert not stale and not unlisted, (
        "[tool.vulture] ignore_names no longer matches what the gate reports.\n"
        f"  Stale -- listed but no longer reported, so the entry now only suppresses "
        f"whatever future symbol reuses the name; delete it: {stale}\n"
        f"  Unlisted -- reported but not listed, so this is either dead code to delete "
        f"or a false positive to add with its reason: {unlisted}"
    )
