"""The shared walk's own contract, since seven gates now read the package through it.

A wrong root here is the failure this walk exists to prevent, arriving once for
every gate instead of once per copy, so the root is pinned as a fact about the
path rather than by whether the walk happens to find files today.
"""

from tests.app_modules import APP_DIR, app_modules


def test_the_walk_finds_the_packages_every_gate_reads_it_for():
    relatives = {relative for relative, _ in app_modules()}
    packages = {relative.split("/")[0] for relative in relatives if "/" in relative}

    assert {"main.py", "config.py", "core/errors.py"} <= relatives
    assert {"api", "audio", "core", "embeddings", "pipeline", "preferences", "stt"} <= packages


def test_the_root_is_the_resolved_package_directory():
    assert APP_DIR.is_absolute()
    assert ".." not in APP_DIR.parts
    assert APP_DIR == APP_DIR.resolve()
    assert (APP_DIR.parent.name, APP_DIR.name) == ("backend", "app")


def test_the_order_is_sorted_and_repeats_across_calls():
    relatives = [relative for relative, _ in app_modules()]
    unsorted_order = [path.relative_to(APP_DIR).as_posix() for path in APP_DIR.rglob("*.py")]

    assert relatives == sorted(relatives)
    assert relatives == [relative for relative, _ in app_modules()]
    assert sorted(unsorted_order) == relatives, (
        "the walk must find exactly what an unordered rglob does, only in a fixed order"
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
