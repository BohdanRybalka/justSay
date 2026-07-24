"""POST /shutdown: token gate, the SIGTERM trigger, and the teardown budget
that must fit inside the Rust wall clock. See
docs/adr/032-production-quit-runs-backend-teardown.md and
specs/049-production-backend-stopped-abruptly/plan.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core import router as core_router
from app.core import tasks
from app.core.config import settings
from app.main import SHUTDOWN_CONNECTION_DRAIN_SECONDS

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"


def _rust_u32_const(name: str) -> int:
    text = BACKEND_RS.read_text(encoding="utf-8")
    match = re.search(rf"const {name}:\s*u32\s*=\s*(\d+);", text)
    assert match, f"could not find `const {name}: u32 = ...;` in {BACKEND_RS}"
    return int(match.group(1))


def _rust_duration_from_millis_const(name: str) -> float:
    text = BACKEND_RS.read_text(encoding="utf-8")
    match = re.search(
        rf"const {name}:\s*Duration\s*=\s*Duration::from_millis\((\d+)\);", text
    )
    assert match, (
        f"could not find `const {name}: Duration = Duration::from_millis(...);` "
        f"in {BACKEND_RS}"
    )
    return int(match.group(1)) / 1000.0


@pytest.fixture
def _token(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "test-secret-token")
    return "test-secret-token"


@pytest.fixture
def _stop_signal_spy(monkeypatch):
    calls: list[None] = []
    monkeypatch.setattr(core_router, "_raise_stop_signal", lambda: calls.append(None))
    return calls


@pytest.mark.anyio
async def test_shutdown_with_valid_token_returns_202_and_raises_signal_once(
    client, _token, _stop_signal_spy
):
    resp = await client.post("/shutdown", headers={"X-JustSay-Token": _token})
    assert resp.status_code == 202
    assert resp.json() == {"status": "stopping"}
    assert len(_stop_signal_spy) == 1


@pytest.mark.anyio
async def test_shutdown_without_token_header_returns_401_and_route_never_ran(
    client, _token, _stop_signal_spy
):
    resp = await client.post("/shutdown")
    assert resp.status_code == 401
    assert _stop_signal_spy == []


@pytest.mark.anyio
async def test_shutdown_with_wrong_token_returns_401_and_route_never_ran(
    client, _token, _stop_signal_spy
):
    resp = await client.post("/shutdown", headers={"X-JustSay-Token": "wrong"})
    assert resp.status_code == 401
    assert _stop_signal_spy == []


@pytest.mark.anyio
async def test_shutdown_with_disallowed_host_returns_400_and_route_never_ran(
    client, _token, _stop_signal_spy
):
    resp = await client.post(
        "/shutdown",
        headers={"X-JustSay-Token": _token, "host": "attacker.example"},
    )
    assert resp.status_code == 400
    assert _stop_signal_spy == []


@pytest.mark.anyio
async def test_shutdown_with_no_token_configured_returns_503_and_other_routes_unaffected(
    client, _stop_signal_spy
):
    assert settings.api_token == ""

    resp = await client.post("/shutdown")
    assert resp.status_code == 503
    assert _stop_signal_spy == []

    resp = await client.get("/history")
    assert resp.status_code == 200


def test_cli_passes_connection_drain_timeout_to_uvicorn(monkeypatch):
    captured: dict = {}

    def _fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", _fake_run)
    monkeypatch.setattr("sys.argv", ["justsay-backend"])

    from app.main import _cli

    _cli()

    assert captured["timeout_graceful_shutdown"] == SHUTDOWN_CONNECTION_DRAIN_SECONDS


def test_backend_teardown_budget_fits_inside_rust_shutdown_endpoint_poll_window():
    poll_max_attempts = _rust_u32_const("SIDECAR_SHUTDOWN_POLL_MAX_ATTEMPTS")
    poll_interval_seconds = _rust_duration_from_millis_const("GRACEFUL_POLL_INTERVAL")
    rust_window = poll_max_attempts * poll_interval_seconds

    backend_worst_case = SHUTDOWN_CONNECTION_DRAIN_SECONDS + tasks.SHUTDOWN_DRAIN_TIMEOUT_SECONDS

    assert backend_worst_case < rust_window, (
        f"backend worst-case teardown ({backend_worst_case}s) must stay strictly below "
        f"the Rust sidecar-shutdown poll window ({rust_window}s), or the graceful poll "
        f"could expire before the backend finishes its own teardown"
    )
