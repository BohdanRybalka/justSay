"""The size gate's exclusion list covers every test file and counts the same from anywhere.

CLAUDE.md → Workflow → Routing decides a task's lane from the lines it changes outside
the test paths (ADR 084). These gates hold the three things that make that number real:
the pathspec drops every test file the repository has, no entry in it has gone stale,
and the answer does not depend on which directory the command was run from.

Unlike the prose pin in test_spec_estimates.py, nothing here reads a gitignored file,
so these run in CI — which is where a newly added test directory first appears.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_TIMEOUT_SECONDS = 30

SIZE_GATE_EXCLUSIONS = (
    "backend/tests",
    "*.test.ts",
    "*.test-helper.ts",
    "vitest.config.ts",
)

_TEST_FILENAME_RE = re.compile(
    r"(?:^|/)(?:"
    r"test_[^/]+\.py"
    r"|[^/]+_test\.py"
    r"|conftest\.py"
    r"|[^/]+\.(?:test|spec)\.(?:ts|tsx|js|jsx)"
    r"|[^/]+\.test-helper\.ts"
    r")$"
)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not on PATH, so no pathspec can be resolved"
)


def _pathspec(*, whole_tree: bool = False) -> list[str]:
    """The exclusion list as git pathspecs, anchored to the repository root.

    `whole_tree` prepends `:(top)`, which `ls-files` needs and `git diff` does not:
    `ls-files` lists only the cwd's subtree unless something says otherwise.
    """
    spec = [f":(top,exclude){pattern}" for pattern in SIZE_GATE_EXCLUSIONS]
    return [":(top)", *spec] if whole_tree else spec


@functools.cache
def _ls_files(args: tuple[str, ...], cwd: str = "") -> tuple[str, ...]:
    """Tracked paths git reports for these pathspecs, run from `cwd` under the repo root.

    `--full-name` keeps the output repository-relative, so two runs from different
    directories are comparable at all.

    Raises `AssertionError` when git refuses, so a broken pathspec fails the gate
    that asked for it instead of emptying the walk and passing.
    """
    done = subprocess.run(
        ["git", "ls-files", "--full-name", "--", *args],
        cwd=REPO_ROOT / cwd if cwd else REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    assert done.returncode == 0, (
        f"git refused the size gate's pathspec {list(args)} from {cwd or '.'}: "
        f"{done.stderr.strip()} — the rule cannot be applied as written"
    )
    return tuple(line for line in done.stdout.splitlines() if line)


def _tracked_test_files() -> tuple[str, ...]:
    """Every tracked file a runner collects, recognised by name rather than by directory."""
    return tuple(path for path in _ls_files(()) if _TEST_FILENAME_RE.search(path))


def test_every_tracked_test_file_is_behind_a_size_gate_exclusion():
    """No separate test file counts toward the lane, whichever directory it appears in."""
    surviving = set(_ls_files(tuple(_pathspec())))
    tests = _tracked_test_files()
    counted = sorted(path for path in tests if path in surviving)

    assert tests, "the walk recognised no test files at all, so this gate proves nothing"
    assert surviving, "the pathspec excluded every tracked file, so this gate proves nothing"
    assert not counted, (
        "these test files still count toward the size gate, so a change is charged for "
        f"testing itself: {counted}"
    )


def test_no_size_gate_exclusion_outlives_the_paths_it_covers():
    """An exclusion matching nothing is a gate quietly widened.

    git accepts any text after `:(exclude)`, so a renamed or invented path is silent:
    the command still runs and the files it was written to drop are counted again.
    """
    tracked = set(_ls_files(()))
    matching_nothing = [
        pattern
        for pattern in SIZE_GATE_EXCLUSIONS
        if set(_ls_files((f":(top,exclude){pattern}",))) == tracked
    ]

    assert tracked, "the walk found no tracked files at all, so this gate proves nothing"
    assert not matching_nothing, (
        "these size-gate exclusions drop no tracked file, so the gate counts what they "
        f"were written to ignore: {matching_nothing}"
    )


def test_the_size_gate_counts_the_same_from_any_directory():
    """The lane must not depend on which directory the command was run from.

    Without `:(top)` the patterns resolve against the shell's cwd, and the workflow
    runs from both the repository root and `backend/`. Measured before the anchor was
    added: the documented command reported 96 production lines from `backend/` and 0
    from the root, which is two different rows of the same table.
    """
    from_root = set(_ls_files(tuple(_pathspec(whole_tree=True))))
    from_backend = set(_ls_files(tuple(_pathspec(whole_tree=True)), "backend"))

    assert from_root, "the pathspec excluded every tracked file, so this gate proves nothing"
    assert from_root == from_backend, (
        "the size gate's pathspec resolves against the working directory, so the same "
        "change measures two different sizes and takes two different lanes. Missing from "
        f"the backend/ run: {sorted(from_root - from_backend)[:5]}; extra: "
        f"{sorted(from_backend - from_root)[:5]}"
    )


def test_the_exclusion_list_admits_it_cannot_see_a_rust_test_module():
    """Rust tests live inside production files, so no pathspec can drop them (ADR 084).

    Pinned rather than left implicit: the day a `tests/` directory appears under
    `src-tauri/`, this fails and the exclusion list gets the entry it now needs.
    """
    rust_test_files = [
        path
        for path in _ls_files(())
        if path.startswith("src-tauri/") and _TEST_FILENAME_RE.search(path)
    ]
    inline = [
        path
        for path in _ls_files(())
        if path.endswith(".rs") and "#[cfg(test)]" in (REPO_ROOT / path).read_text(encoding="utf-8")
    ]

    assert inline, (
        "no Rust file carries `#[cfg(test)]` any more, so the limit ADR 084 records no "
        "longer exists and the ADR should say so"
    )
    assert not rust_test_files, (
        "Rust test files now live in their own paths, which the exclusion list can cover "
        f"and does not: {rust_test_files}"
    )
