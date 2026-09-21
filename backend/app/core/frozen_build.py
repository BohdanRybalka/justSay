"""Whether this process is a PyInstaller bundle rather than a source checkout.

The flag is read at call time, never captured at import (ADR 090).
"""

import sys


def is_frozen_build() -> bool:
    """True in a PyInstaller bundle, False in a source checkout.

    Reads the bootloader flag alone: no environment variable and no `_MEIPASS`
    path takes part in the answer.
    """
    return bool(getattr(sys, "frozen", False))
