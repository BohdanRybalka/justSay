"""The one doorway from `core` to the composition root above it.

`AppSettings` and the `settings` singleton are defined in `app/config.py`, at
the package root, because assembling them means importing every feature
package's settings class. This module re-exports them so that `core` — the
layer every package may import — contains no feature-package import at all,
while every call site keeps spelling the singleton `app.core.config.settings`.

The re-export binds the same objects rather than constructing new ones, so
`app.core.config.settings is app.config.settings`. It is permanent, not a
migration shim: see ADR 076 for why deleting it puts a package cycle back.
"""

from app.config import AppSettings, settings

__all__ = ["AppSettings", "settings"]
