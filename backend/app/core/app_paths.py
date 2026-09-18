"""Single source of truth for where ``settings.json``, ``history.db`` and
``backend.log`` live on disk, and for the dev/production split between them
(ADR 012).

One path deliberately does not come from here: the whisper.cpp model cache is
spelled ``~/.justsay/models`` outright and never passes through the dev split,
so a from-source run shares its downloads with the installed app.
"""

import os
import sys
from pathlib import Path

_DATA_DIR_ENV_VAR = "JUSTSAY_DATA_DIR"
_FORCE_DEV_ENV_VAR = "JUSTSAY_FORCE_DEV_DATA_DIR"

PROD_DIR_NAME = ".justsay"
DEV_DIR_NAME = ".justsay-dev"
TEMP_DIR_NAME = "tmp"


def resolve_app_data_root() -> Path:
    """The data directory root: ``settings.json``, ``history.db`` and ``backend.log``.

    First match wins: ``JUSTSAY_DATA_DIR``; else ``~/.justsay`` when ``sys.frozen`` is set and
    ``JUSTSAY_FORCE_DEV_DATA_DIR`` is not; else ``~/.justsay-dev``, never the user's real history.
    """
    override = os.environ.get(_DATA_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    is_frozen = getattr(sys, "frozen", False)
    forced_dev = os.environ.get(_FORCE_DEV_ENV_VAR) == "1"
    if is_frozen and not forced_dev:
        return Path.home() / PROD_DIR_NAME
    return Path.home() / DEV_DIR_NAME


def resolve_temp_dir() -> Path:
    """The scratch directory for recorded and uploaded audio.

    The single definition of that path, so the directory the cleanup endpoint empties and the
    directory the ``output_dir`` validator refuses to store history in cannot drift (ADR 033).
    """
    return resolve_app_data_root() / TEMP_DIR_NAME
