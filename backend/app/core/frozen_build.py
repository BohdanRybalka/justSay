"""Whether this process is a PyInstaller bundle rather than a source checkout.

The flag is read at call time, so a test that sets or deletes it on `sys` is
seen by every caller (ADR 090).
"""

import sys


def is_frozen_build() -> bool:
    """True in a PyInstaller bundle, False in a source checkout.

    Reads the bootloader flag alone: no environment variable and no `_MEIPASS`
    path takes part in the answer.
    """
    return bool(getattr(sys, "frozen", False))
