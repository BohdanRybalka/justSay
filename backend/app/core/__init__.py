"""Shared primitives every other package may import.

The rule for this package: no module here imports a feature package. There is
no exception to it and `tests/test_import_layers.py` fails on one. `types`,
`constants`, `app_paths`, `utils`, `tasks`, `logging_config`, `gpu_probe`,
`schemas`, `auth_middleware`, `audio_formats` and `errors` are what that rule
leaves — primitives any package can reach without acquiring anything below it.
`errors` holds the `JustSayError` hierarchy and is framework-free for exactly
that reason.

One module reaches *upward* instead, and only one is allowed to:

- `config.py` is the single doorway to the composition root. `AppSettings` and
  the `settings` singleton are defined in `app/config.py`, which sits above
  every package because assembling them means importing every package's own
  `*Settings` class. `config.py` re-exports them, so callers keep finding
  `settings` where they always have while `core` itself stays a leaf. A second
  module here reaching for `app.config` fails a test. See ADR 076.

Two modules here are HTTP-boundary code rather than primitives:

- `router.py` serves the operational endpoints (`/health`, `/shutdown`).
- `error_handler.py` turns a `JustSayError` into its HTTP response. It is an
  HTTP-boundary module like `router.py`, and it lives here rather than beside
  `errors.py`'s callers because the registration belongs to the app itself.

Until spec 076 this package also held the transcript store, the user
preferences and four HTTP routers, which made it simultaneously above and below
the feature packages; roughly half the function-local imports in the backend
existed to defer around the resulting cycles. Those modules now live in
`app.transcripts` and `app.preferences`. Spec 165 removed the last of it: until
then `config.py` imported `app.audio`, `app.stt` and `app.embeddings` from
inside `core`, which was three of the four tolerated package cycles. ADR 044 is
the dissolution this completes; ADR 076 is the arrangement that replaced it.
"""
