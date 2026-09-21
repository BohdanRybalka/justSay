"""Tests for `app.core.frozen_build.is_frozen_build` -- the one derivation of the
PyInstaller bootloader flag, and the walk that reddens when a second one appears
(docs/adr/090-one-answer-to-whether-this-is-a-packaged-build.md).

Mirrors test_app_paths.py's monkeypatch.setattr(sys, "frozen", ...) /
monkeypatch.delattr(sys, "frozen", raising=False) conventions for simulating the
bootloader flag, which no test in this repository can set for real.
"""

import ast
import sys
from pathlib import Path

import pytest

from app.core.frozen_build import is_frozen_build

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_PRIMITIVE_MODULE = "core/frozen_build.py"


@pytest.fixture(autouse=True)
def _no_bootloader_flag(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)




def test_frozen_attribute_set_reports_a_packaged_build(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert is_frozen_build() is True




def test_absent_frozen_attribute_reports_a_source_checkout():
    assert is_frozen_build() is False




def test_falsy_frozen_attribute_reports_a_source_checkout(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert is_frozen_build() is False




def test_truthy_non_boolean_frozen_attribute_is_narrowed_to_a_bool(monkeypatch):
    monkeypatch.setattr(sys, "frozen", "console_exe", raising=False)

    assert is_frozen_build() is True




def test_forcing_the_dev_data_dir_does_not_make_a_packaged_build_a_checkout(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("JUSTSAY_FORCE_DEV_DATA_DIR", "1")

    assert is_frozen_build() is True




def _app_module_trees() -> tuple[tuple[str, ast.Module], ...]:
    """Every module under `backend/app` as (relative posix path, parsed tree)."""
    return tuple(
        (path.relative_to(_APP_DIR).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(_APP_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _derives_the_bootloader_flag(tree: ast.Module) -> bool:
    """Whether a module reads `sys.frozen` itself, by getattr or by attribute access.

    Walks the syntax tree rather than the text, so a docstring naming the flag in
    prose is not a derivation.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "frozen":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                return True
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        target, attribute = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "sys"):
            continue
        if isinstance(attribute, ast.Constant) and attribute.value == "frozen":
            return True
    return False


def test_one_module_alone_derives_the_bootloader_flag():
    trees = _app_module_trees()
    derivations = sorted(module for module, tree in trees if _derives_the_bootloader_flag(tree))

    assert trees
    assert derivations == [_PRIMITIVE_MODULE], (
        "the packaged-build question has one answer; call "
        f"app.core.frozen_build.is_frozen_build() instead: {derivations}"
    )
