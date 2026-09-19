"""Where a `JustSayError` becomes an HTTP response — the half of `errors` that knows about HTTP.

The handler is registered on the base class alone; Starlette walks `type(exc).__mro__`, so every
subclass is covered by that one registration, and a subclass needing its own treatment can be
registered separately.

Out of reach: an exception raised inside a middleware, and one raised from a background task after
the response has started. Neither is a refusal the user can be shown.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.errors import JustSayError

log = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    """Every refusal answers with exactly these two keys.

    `detail` is the user-facing sentence the frontend renders. `code` is the refusal's stable
    machine-readable name, a literal on the error class, so renaming a class cannot change it.
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
