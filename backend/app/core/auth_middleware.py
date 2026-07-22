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


class LaunchTokenMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # OPTIONS is the CORS preflight (already short-circuited by CORS, but
        # exempt here too); /health is the Rust watchdog's readiness poll,
        # which reqwest sends without a token.
        if scope["method"] == "OPTIONS" or scope["path"] == "/health":
            await self.app(scope, receive, send)
            return

        # Read at request time so tests can monkeypatch settings.api_token and
        # so a manually started backend with no token stays open (dev workflow).
        expected = settings.api_token
        if not expected:
            await self.app(scope, receive, send)
            return

        provided = b""
        for name, value in scope["headers"]:
            if name == _TOKEN_HEADER:
                provided = value
                break

        # Bytes comparison so a non-ASCII forged header can't raise inside
        # compare_digest; constant-time to avoid leaking the token by timing.
        if not secrets.compare_digest(provided, expected.encode("utf-8")):
            response = JSONResponse(
                {"detail": "Missing or invalid API token"}, status_code=401
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
