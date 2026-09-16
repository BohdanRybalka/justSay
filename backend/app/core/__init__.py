"""Shared primitives every other package may import.

The rule for this package: no module here imports a feature package. There is
no exception to it and `tests/test_import_layers.py` fails on one. `types`,
`constants`, `app_paths`, `utils`, `tasks`, `logging_config`, `gpu_probe`,
`audio_formats`, `schemas` and `errors` are what that rule leaves — primitives
any package can reach without acquiring anything below them. `errors` holds the
`JustSayError` hierarchy and is framework-free for exactly that reason.

One module reaches *upward* instead, and only one is allowed to:

- `config.py` is the single doorway to the composition root. `AppSettings` and
  the `settings` singleton are defined in `app/config.py`, which sits above
  every package because assembling them means importing every package's own
  `*Settings` class. `config.py` re-exports them, so callers keep finding
  `settings` where they always have. The cost is that `core` is not a leaf and
  this arrangement does not make it one: importing `app.core.config` loads
  every feature package's settings module through the root above. A second
  module here reaching for `app.config` fails a test. See ADR 076.

Three modules here are HTTP-boundary code rather than primitives:

- `router.py` serves the operational endpoints (`/health`, `/shutdown`).
- `error_handler.py` turns a `JustSayError` into its HTTP response. It lives
  here rather than beside `errors.py`'s callers because the registration
  belongs to the app itself.
- `auth_middleware.py` is the pure-ASGI token gate (ADR 026). It imports
  `starlette` and reads the `settings` singleton through `config.py`, so it
  acquires the whole settings graph and is not a primitive either.

Until spec 076 this package also held the transcript store, the user
preferences and four HTTP routers, which made it simultaneously above and below
the feature packages; roughly half the function-local imports in the backend
existed to defer around the resulting cycles. Those modules now live in
`app.transcripts` and `app.preferences`. Spec 165 finished the rule rather than
the tangle: `config.py` imported `app.audio`, `app.stt` and `app.embeddings`
from inside `core` until then, and moving that assembly up to `app/config.py`
left the loops running one hop longer through it. ADR 044 is the dissolution
this continues; ADR 076 is the arrangement that replaced the old exception list
and records what it did and did not buy.
"""
