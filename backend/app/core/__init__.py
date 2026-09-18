"""Shared primitives every other package may import.

The rule: no module here imports a feature package, and
`tests/test_import_layers.py` fails on one. `config.py` is the single doorway
up to the composition root, and `router.py`, `error_handler.py` and
`auth_middleware.py` are HTTP-boundary code rather than primitives — which is
why this package is not a leaf (ADR 076).
"""
