"""Shared primitives every other package may import.

The rule for this package: a module here imports from `app` only if it is one
of the three documented exceptions below. Everything else — `types`,
`constants`, `app_paths`, `utils`, `tasks`, `logging_config`, `gpu_probe`,
`schemas`, `auth_middleware`, `audio_formats`, `errors` — depends on nothing
inside `app`, which is what makes it safe for any package to reach. `errors`
holds the `JustSayError` hierarchy and is framework-free for exactly that
reason.

Three modules deliberately break that rule:

- `config.py` is the composition root. It assembles `AppSettings` out of every
  package's own `*Settings` class, so it necessarily imports `app.audio`,
  `app.stt`, `app.llm` and `app.embeddings`. It sits *above* every package
  rather than beneath them, and lives here only because that is where callers
  already look for `settings`. See ADR 044.
- `router.py` serves the operational endpoints (`/health`, `/shutdown`).
- `error_handler.py` turns a `JustSayError` into its HTTP response. It is an
  HTTP-boundary module like `router.py`, and it lives here rather than beside
  `errors.py`'s callers because the registration belongs to the app itself.

Until spec 076 this package also held the transcript store, the user
preferences and four HTTP routers, which made it simultaneously above and below
the feature packages; roughly half the function-local imports in the backend
existed to defer around the resulting cycles. Those modules now live in
`app.transcripts` and `app.preferences`.
"""
