"""The one walk over `backend/app`, shared by every gate that scans the package.

Each gate needs the same three answers -- where the package is, which files are
its modules, and how one is spelled relative to the package -- and every root a
caller spells for itself is a way for two gates to disagree about what they
cover. The exclusions stay with the caller: one excludes by package, one by an
allowlist of labels, one by nothing at all, and folding them together here
would silently change what each gate checks.
"""

import functools
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"


@functools.cache
def app_modules() -> tuple[tuple[str, Path], ...]:
    """Every module under `APP_DIR`, sorted, as (path relative to it, absolute path).

    Asserts its own result non-empty: one blind walk here would leave every
    gate sharing it passing over nothing at all.
    """
    found = tuple(
        (path.relative_to(APP_DIR).as_posix(), path)
        for path in sorted(APP_DIR.rglob("*.py"))
    )
    assert found, f"no module found under {APP_DIR}"
    return found
