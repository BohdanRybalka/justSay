"""Single source of truth for where ``settings.json``, ``history.db``, and
``backend.log`` live on disk.

``user_settings.py`` and ``history.py`` both need this resolution logic, but
``history.py`` deliberately does not import ``user_settings`` (a one-way
dependency documented in ``history.py``'s own module docstring) -- so it
lives here, one level below both, instead of being owned by either. See
``docs/adr/012-dev-mode-data-directory-isolation.md`` for the full rationale.
"""

import os
import sys
from pathlib import Path

_DATA_DIR_ENV_VAR = "JUSTSAY_DATA_DIR"
_FORCE_DEV_ENV_VAR = "JUSTSAY_FORCE_DEV_DATA_DIR"

PROD_DIR_NAME = ".justsay"
DEV_DIR_NAME = ".justsay-dev"


def resolve_app_data_root() -> Path:
    """Resolution order (first match wins):

    1. ``JUSTSAY_DATA_DIR`` env override -- an explicit full-path escape hatch.
    2. ``sys.frozen`` (the PyInstaller bootloader flag, already used in this
       codebase by ``app.stt.local_setup`` and
       ``app.stt.local_vulkan_cmd::resolve_binary_path`` for the identical
       "packaged binary vs. running from source" distinction) is ``True``
       **and** ``JUSTSAY_FORCE_DEV_DATA_DIR`` is not set -> ``~/.justsay``
       (production, byte-identical to today's default).
    3. Otherwise -> ``~/.justsay-dev``.

    Case 3 is the default for every from-source invocation -- dev is the
    fallback, not production, so an unrecognized launch context never writes
    into the user's real history.
    """
    override = os.environ.get(_DATA_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    is_frozen = getattr(sys, "frozen", False)
    forced_dev = os.environ.get(_FORCE_DEV_ENV_VAR) == "1"
    if is_frozen and not forced_dev:
        return Path.home() / PROD_DIR_NAME
    return Path.home() / DEV_DIR_NAME
