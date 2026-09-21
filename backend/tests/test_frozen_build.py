"""Tests for `app.core.frozen_build.is_frozen_build` and for the walk that reddens
when a second module derives the PyInstaller bootloader flag (ADR 090).
"""

import ast
import sys

import pytest

from app.core.frozen_build import is_frozen_build
from tests.app_modules import APP_DIR, app_modules

_PRIMITIVE_MODULE = "core/frozen_build.py"
_FORCE_DEV_MODULE = "core/app_paths.py"
_FORCE_DEV_FLAG = "JUSTSAY_FORCE_DEV_DATA_DIR"


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
    monkeypatch.setenv(_FORCE_DEV_FLAG, "1")

    assert is_frozen_build() is True


def _app_module_trees() -> tuple[tuple[str, ast.Module], ...]:
    """Every module under `backend/app` as (relative posix path, parsed tree)."""
    return tuple(
        (relative, ast.parse(path.read_text(encoding="utf-8")))
        for relative, path in app_modules()
    )


def _derives_the_bootloader_flag(tree: ast.Module) -> bool:
    """Whether a module reads `sys.frozen` itself, by attribute, `getattr` or `hasattr`.

    Walks the syntax tree rather than the text, so a docstring naming the flag in
    prose is not a derivation. An aliased import or a computed attribute name
    passes unseen.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "frozen":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                return True
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr")):
            continue
        if len(node.args) < 2:
            continue
        target, attribute = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "sys"):
            continue
        if isinstance(attribute, ast.Constant) and attribute.value == "frozen":
            return True
    return False


def _names_the_force_dev_flag(tree: ast.Module) -> bool:
    """Whether a module spells the force-dev environment variable anywhere at all."""
    return any(
        isinstance(node, ast.Constant) and node.value == _FORCE_DEV_FLAG
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "source",
    [
        "if sys.frozen: pass",
        'x = getattr(sys, "frozen", False)',
        'x = hasattr(sys, "frozen")',
    ],
)
def test_every_spelling_the_walk_claims_to_see_is_seen(source):
    """Each branch of the detector, exercised on its own spelling.

    No module under `backend/app` writes the bare-attribute or `hasattr` forms, so
    the real-tree walk below leaves both branches unrun."""
    assert _derives_the_bootloader_flag(ast.parse(source)) is True


def test_naming_the_flag_in_prose_is_not_a_derivation():
    source = '"""Reads sys.frozen and the frozen attribute of sys."""'

    assert _derives_the_bootloader_flag(ast.parse(source)) is False


def test_one_module_alone_derives_the_bootloader_flag():
    trees = _app_module_trees()
    derivations = sorted(module for module, tree in trees if _derives_the_bootloader_flag(tree))
    offenders = [module for module in derivations if module != _PRIMITIVE_MODULE]

    assert trees, f"the walk found no module under {APP_DIR} to read"
    assert derivations == [_PRIMITIVE_MODULE], (
        "the packaged-build question has one answer; call "
        f"app.core.frozen_build.is_frozen_build() instead of deriving it in: {offenders}"
    )


def test_one_module_alone_names_the_force_dev_flag():
    """The flag pairs with frozen-ness in one reader; the primitive must stay free of it.

    Folding it into `is_frozen_build()` would make a packaged build with the flag set
    claim it has an interpreter to pip into."""
    trees = _app_module_trees()
    readers = sorted(module for module, tree in trees if _names_the_force_dev_flag(tree))

    assert trees, f"the walk found no module under {APP_DIR} to read"
    assert readers == [_FORCE_DEV_MODULE], (
        f"{_FORCE_DEV_FLAG} belongs to the one reader that pairs it with frozen-ness: {readers}"
    )
