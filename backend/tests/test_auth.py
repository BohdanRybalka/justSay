"""Loopback API authentication: Host allowlist + per-launch token gate.

See docs/adr/026-loopback-api-request-authentication.md and
specs/040-local-api-web-exposure/plan.md.
"""

from __future__ import annotations

import pytest

from app.core import user_settings
from app.core.auth_middleware import _EXEMPT_PATHS
from app.core.config import settings


# --- TrustedHostMiddleware: Host-header allowlist ---------------------------


@pytest.mark.anyio
async def test_disallowed_host_is_rejected_with_400_before_the_route(client):
    """A rebound Host outside the allowlist is rejected with 400 before any
    route runs -- verified on a read (GET /history) and a write (PUT /settings)."""
    resp = await client.get("/history", headers={"host": "attacker.example"})
    assert resp.status_code == 400

    resp = await client.put(
        "/settings",
        json={"shortcut": "Ctrl+Alt+KeyZ"},
        headers={"host": "attacker.example"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:9377", "localhost", "localhost:9377"])
async def test_allowed_host_with_or_without_port_passes(client, host):
    """An allowed Host (127.0.0.1 / localhost, with or without :9377) passes
    the host check. No token is configured, so the request completes normally."""
    resp = await client.get("/health", headers={"host": host})
    assert resp.status_code == 200


# --- LaunchTokenMiddleware: token gate --------------------------------------


@pytest.fixture
def _token(monkeypatch):
    """Configure a per-launch token on the runtime settings singleton."""
    monkeypatch.setattr(settings, "api_token", "test-secret-token")
    return "test-secret-token"


@pytest.mark.anyio
async def test_protected_get_without_token_returns_401(client, _token):
    resp = await client.get("/history")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_protected_get_with_correct_token_returns_2xx(client, _token):
    resp = await client.get("/history", headers={"X-JustSay-Token": _token})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_protected_post_without_token_returns_401_and_route_never_runs(client, _token):
    """POST /audio/start without the token is rejected by the middleware before
    the route runs -- so no recorder is touched (the ASGI client never ran the
    lifespan that creates app.state.recorder; a 401 proves the route body was
    never reached)."""
    resp = await client.post("/audio/start")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_put_settings_without_token_does_not_mutate(client, _token):
    """PUT /settings without the token returns 401 and the side effect (a
    settings mutation) does not occur."""
    before = user_settings.get_user_settings().shortcut

    resp = await client.put("/settings", json={"shortcut": "Ctrl+Alt+KeyZ"})
    assert resp.status_code == 401

    assert user_settings.get_user_settings().shortcut == before


@pytest.mark.anyio
async def test_wrong_token_returns_401(client, _token):
    resp = await client.get("/history", headers={"X-JustSay-Token": "wrong"})
    assert resp.status_code == 401


# --- Exemptions: /health and OPTIONS ----------------------------------------


def test_exempt_paths_are_exactly_health():
    """The exempt-path set is duplicated in the frontend as TOKEN_EXEMPT_PATHS
    (src/api.ts), which uses it to decide that a 2xx from such a path proves
    nothing about authentication. Exempting a second path here without
    mirroring it there would let that path's 200 clear the frontend's auth
    flag and repaint the status badge green over an unauthorized app -- the
    exact failure ADR 028 exists to remove. Update both, or neither."""
    assert _EXEMPT_PATHS == frozenset({"/health"})


@pytest.mark.anyio
async def test_health_is_exempt_from_the_token_check(client, _token):
    """GET /health returns 200 without any token even when one is configured,
    so the Rust watchdog's readiness poll keeps working."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_options_preflight_is_not_rejected_by_the_token_check(client, _token):
    """An OPTIONS preflight to a protected endpoint is not 401'd -- it receives
    the CORS preflight response so the WebView's cross-origin requests complete."""
    resp = await client.options(
        "/history",
        headers={
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code != 401
    assert resp.status_code == 200


# --- Open mode: no token configured -----------------------------------------


@pytest.mark.anyio
async def test_open_mode_no_token_configured_serves_protected_endpoints(client):
    """With no token configured (settings.api_token == ""), protected endpoints
    return their normal 2xx without any token header -- the manual dev workflow
    is untouched."""
    assert settings.api_token == ""

    resp = await client.get("/history")
    assert resp.status_code == 200

    resp = await client.put("/settings", json={"shortcut": "Ctrl+Alt+KeyM"})
    assert resp.status_code == 200
