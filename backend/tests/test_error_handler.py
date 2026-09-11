"""What a refusal looks like on the wire, asserted key by key.

The app under test is built by the real `register_error_handlers`, not by a
hand-registered handler, so the registration itself is covered rather than
assumed. The body is compared as a whole dict: a third key appearing later
fails these tests instead of quietly reaching the frontend.

Mutations actually run against `app/core/error_handler.py`, with the number of
tests in this file each one reddens:

- `register_error_handlers` made a no-op -- six, every test but the
  `RuntimeError` pass-through
- `ErrorBody` given a third field defaulting to `None` -- six
- `exc.diagnostic` appended to `detail` -- six
- `headers=` dropped from the `JSONResponse` -- one
- the handler registered on `Exception` instead of `JustSayError` -- all seven,
  because it then swallows the `RuntimeError` too
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.error_handler import register_error_handlers
from app.core.errors import (
    ConfigurationError,
    JustSayError,
    NotReadyError,
    ResourceUnavailableError,
)

_MESSAGE = "Add your Gemini API key in Settings."
_DIAGNOSTIC = "provider replied 401 invalid_api_key on model gemini-2.5-flash"


def _app_raising(exc: Exception) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    return app


@pytest.mark.parametrize(
    ("factory", "status", "code"),
    [
        (ConfigurationError, 400, "configuration_error"),
        (ResourceUnavailableError, 503, "resource_unavailable"),
        (NotReadyError, 409, "not_ready"),
    ],
)
def test_each_subclass_answers_its_own_status_and_exact_body(
    factory: type[JustSayError], status: int, code: str
) -> None:
    with TestClient(_app_raising(factory(_MESSAGE))) as client:
        response = client.get("/boom")
    assert response.status_code == status
    assert response.json() == {"detail": _MESSAGE, "code": code}


def test_the_diagnostic_reaches_the_log_and_never_the_wire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = ConfigurationError(_MESSAGE, diagnostic=_DIAGNOSTIC)
    with caplog.at_level(logging.INFO, logger="app.core.error_handler"):
        with TestClient(_app_raising(exc)) as client:
            response = client.get("/boom")
    body = response.json()
    assert set(body) == {"detail", "code"}
    assert _DIAGNOSTIC not in response.text
    assert body["detail"] == _MESSAGE
    assert any(_DIAGNOSTIC in record.getMessage() for record in caplog.records)


def test_a_refusal_carries_the_headers_it_was_given() -> None:
    exc = ResourceUnavailableError("Transcript store busy", headers={"Retry-After": "1"})
    with TestClient(_app_raising(exc)) as client:
        response = client.get("/boom")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json() == {"detail": "Transcript store busy", "code": "resource_unavailable"}


def test_a_subclass_of_a_subclass_is_caught_by_the_one_registration() -> None:
    """Starlette resolves the handler by walking the MRO, so depth is free."""

    class DeviceHeldElsewhereError(ResourceUnavailableError):
        pass

    with TestClient(_app_raising(DeviceHeldElsewhereError("The microphone is in use."))) as client:
        response = client.get("/boom")
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The microphone is in use.",
        "code": "resource_unavailable",
    }


def test_a_plain_runtime_error_is_left_alone() -> None:
    """The handler claims refusals only; a crash stays a crash."""
    with TestClient(_app_raising(RuntimeError("something broke"))) as client:
        with pytest.raises(RuntimeError, match="something broke"):
            client.get("/boom")
