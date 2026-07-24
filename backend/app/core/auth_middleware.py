"""Per-launch shared-secret gate for the loopback API.

Rejects any request whose ``X-JustSay-Token`` header does not match the
per-launch token in ``settings.api_token`` with ``401``. See
docs/adr/026-loopback-api-request-authentication.md.

Deliberately a **pure-ASGI** middleware, NOT ``BaseHTTPMiddleware`` /
``@app.middleware("http")``: ``BaseHTTPMiddleware`` buffers the response body,
which would break the ``StreamingResponse`` SSE endpoints
(``GET /audio/level-stream`` and ``POST /stt/local/install``).
"""

import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings

_TOKEN_HEADER = b"x-justsay-token"

_EXEMPT_PATHS = frozenset({"/health"})


class LaunchTokenMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["method"] == "OPTIONS" or scope["path"] in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        expected = settings.api_token
        if not expected:
            await self.app(scope, receive, send)
            return

        provided = b""
        for name, value in scope["headers"]:
            if name == _TOKEN_HEADER:
                provided = value
                break

        if not secrets.compare_digest(provided, expected.encode("utf-8")):
            response = JSONResponse(
                {"detail": "Missing or invalid API token"}, status_code=401
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
