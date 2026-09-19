"""Shared primitives every other package may import.

The rule: no module here imports a feature package or the HTTP boundary, and no
module here imports a web framework at all — `tests/test_import_layers.py` fails
on any of the three. `config.py` is the single doorway up to the composition
root, which is why this package is not a leaf (ADR 076).
"""
