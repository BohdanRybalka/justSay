"""Tests for `app.core.app_paths.resolve_app_data_root` -- the resolution
order documented in docs/adr/012-dev-mode-data-directory-isolation.md.

Mirrors test_local_vulkan_cmd.py's monkeypatch.setattr(sys, "frozen", ...) /
monkeypatch.delattr(sys, "frozen", raising=False) / monkeypatch.setenv/delenv
conventions for simulating the PyInstaller-frozen bootloader flag.
"""

import sys
from pathlib import Path

import pytest

from app.core import app_paths
from app.core.app_paths import resolve_app_data_root


@pytest.fixture(autouse=True)
def _isolated_home_and_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("JUSTSAY_DATA_DIR", raising=False)
    monkeypatch.delenv("JUSTSAY_FORCE_DEV_DATA_DIR", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    yield home




def test_defaults_to_dev_dir_when_not_frozen_and_no_env_vars(_isolated_home_and_env):
    home = _isolated_home_and_env
    assert resolve_app_data_root() == home / ".justsay-dev"




def test_frozen_and_not_forced_dev_resolves_to_prod_dir(_isolated_home_and_env, monkeypatch):
    home = _isolated_home_and_env
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert resolve_app_data_root() == home / ".justsay"




def test_frozen_and_forced_dev_resolves_to_dev_dir(_isolated_home_and_env, monkeypatch):
    home = _isolated_home_and_env
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("JUSTSAY_FORCE_DEV_DATA_DIR", "1")

    assert resolve_app_data_root() == home / ".justsay-dev"




@pytest.mark.parametrize(
    "is_frozen, forced_dev",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_data_dir_override_wins_regardless_of_frozen_or_forced_dev(
    tmp_path, monkeypatch, is_frozen, forced_dev
):
    override = tmp_path / "custom-data-dir"
    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(override))
    if forced_dev:
        monkeypatch.setenv("JUSTSAY_FORCE_DEV_DATA_DIR", "1")
    if is_frozen:
        monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert resolve_app_data_root() == override


def test_data_dir_override_expands_user(monkeypatch):
    """Path.expanduser() reads the real OS home (USERPROFILE/HOME env var),
    not the Path.home() classmethod -- the autouse fixture's Path.home patch
    does not affect it, so this compares against Path.expanduser() directly
    rather than the fixture's fake home."""
    monkeypatch.setenv("JUSTSAY_DATA_DIR", "~/custom-data-dir")

    result = resolve_app_data_root()

    assert result == Path("~/custom-data-dir").expanduser()
    assert "~" not in str(result)




def test_dir_name_constants_are_distinct():
    assert app_paths.PROD_DIR_NAME == ".justsay"
    assert app_paths.DEV_DIR_NAME == ".justsay-dev"
    assert app_paths.PROD_DIR_NAME != app_paths.DEV_DIR_NAME
