"""The one doorway from `core` to the composition root above it.

`AppSettings` and the `settings` singleton are defined in `app/config.py`,
because assembling them means importing every feature package's settings
class. This module re-exports them, binding the same objects rather than
constructing new ones, so `app.core.config.settings is app.config.settings`
and `core` itself contains no feature-package import. It is permanent, not a
migration shim: deleting it puts a package cycle back (ADR 076).
"""

from app.config import AppSettings, settings

__all__ = ["AppSettings", "settings"]
