"""The one walk over `backend/app`, shared by every gate that scans the package.

A gate that spells the root for itself is a way for two gates to disagree about
what the package is. Exclusions stay with the caller: they are not
interchangeable, and folding them together here would change what each gate
covers.
"""

import functools
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"


@functools.cache
def app_modules() -> tuple[tuple[str, Path], ...]:
    """Every module under `APP_DIR`, sorted by its POSIX relative spelling.

    Yields `(path relative to APP_DIR, absolute path)`. Asserts its own result
    non-empty: one blind walk here would leave every gate sharing it passing
    over nothing at all.
    """
    found = tuple(
        sorted(
            ((path.relative_to(APP_DIR).as_posix(), path) for path in APP_DIR.rglob("*.py")),
            key=lambda module: module[0],
        )
    )
    assert found, f"no module found under {APP_DIR}"
    return found
