"""The one place a `JustSayError` becomes an HTTP response.

`app/core/errors.py` stays framework-free so every layer can raise from it;
this module is the half that knows about HTTP, which is why it is the only
module in `core` besides `router.py` and `auth_middleware.py` that
`tests/test_import_layers.py` lets import a web framework.

The handler is registered on the base class alone. Starlette's
`_lookup_exception_handler` walks `type(exc).__mro__` and takes the first
registered class it finds, so every subclass — including ones added later — is
covered by that single registration, and a subclass that ever needs its own
treatment can be registered separately and wins by MRO order.

Two shapes are deliberately outside its reach. An exception raised inside a
middleware never reaches `ExceptionMiddleware`, which Starlette appends as the
innermost layer, and an exception raised from a background task surfaces after
the response has started, where Starlette turns it into
`RuntimeError("Caught handled exception, but response already started.")`.
Neither is a refusal the user can be shown.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.errors import JustSayError

log = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    """Every refusal answers with exactly these two keys.

    `detail` is kept because the frontend already reads it and falls back to
    `HTTP <status>` without it. `code` is the stable machine-readable name of
    the refusal — a literal declared on the class, never derived from
    `type(exc).__name__`, so renaming a class cannot change what a client sees.
    """

    detail: str
    code: str


async def justsay_error_handler(request: Request, exc: JustSayError) -> JSONResponse:
    """Log the refusal once, with its diagnostic, and answer without it."""
    log.info(
        "Refused %s %s -> %s %s: %s | diagnostic=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.code,
        exc.message,
        exc.diagnostic,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorBody(detail=exc.message, code=exc.code).model_dump(),
        headers=dict(exc.headers) if exc.headers else None,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Wire the hierarchy into the app, once, on the base class."""
    app.add_exception_handler(JustSayError, justsay_error_handler)
