"""The shared walk's own contract, since seven gates now read the package through it.

A wrong root here is the failure this walk exists to prevent, arriving once for
every gate instead of once per copy, so the root is pinned as a fact about the
path rather than by whether the walk happens to find files today.
"""

import inspect

from tests import app_modules as app_modules_module
from tests.app_modules import APP_DIR, app_modules

_PACKAGES = {
    "api",
    "audio",
    "core",
    "embeddings",
    "pipeline",
    "preferences",
    "stt",
    "transcripts",
}


def test_the_walk_finds_the_packages_it_is_meant_to_check():
    relatives = {relative for relative, _ in app_modules()}
    packages = {relative.split("/")[0] for relative in relatives if "/" in relative}

    assert {"main.py", "config.py", "core/errors.py"} <= relatives
    assert packages == _PACKAGES, (
        "a package that appears or disappears changes what every gate sharing this walk "
        f"covers, and nothing else would say so: {sorted(packages ^ _PACKAGES)}"
    )


def test_the_root_is_the_resolved_package_directory():
    source = inspect.getsource(app_modules_module)

    assert APP_DIR.is_absolute()
    assert ".." not in APP_DIR.parts
    assert APP_DIR == APP_DIR.resolve()
    assert (APP_DIR.parent.name, APP_DIR.name) == ("backend", "app")
    assert "Path(__file__).resolve()" in source, (
        "the assertions above hold on any checkout reached without a symlink, so the "
        "resolve() is pinned as source text too: one of the seven copies this walk "
        "replaced had none, and a junction is where that difference shows"
    )


def test_the_order_is_sorted_and_repeats_across_calls():
    relatives = [relative for relative, _ in app_modules()]
    unsorted_order = [path.relative_to(APP_DIR).as_posix() for path in APP_DIR.rglob("*.py")]

    assert relatives == sorted(relatives)
    assert sorted(unsorted_order) == relatives, (
        "the walk must find exactly what an unordered rglob does, only in a fixed order"
    )
    assert app_modules() is app_modules(), (
        "every gate calls this, most of them more than once; without the cache the "
        "package is re-walked each time and a fresh tuple is handed back"
    )


def test_each_relative_path_is_posix_spelled_and_rejoins_its_own_file():
    relatives = [relative for relative, _ in app_modules()]
    backslashed = [relative for relative in relatives if "\\" in relative]
    misjoined = [
        relative for relative, path in app_modules() if APP_DIR / relative != path
    ]

    assert any("/" in relative for relative in relatives), (
        "a walk reaching no sub-package would make the separator assertion vacuous"
    )
    assert not backslashed, f"these are spelled the platform's way, not POSIX: {backslashed}"
    assert not misjoined, f"these do not rejoin the file they were taken from: {misjoined}"
