"""The difference between a refusal the user should read and a bug.

A `JustSayError` means the backend declined to do something and can say why in
a sentence a person understands. Anything else — an invariant that broke, a
library that misbehaved, a device that failed mid-use — is not a member of this
hierarchy and keeps propagating into a 500, which is what a crash should look
like. That membership is the whole line: `except JustSayError` now states it,
where matching on a substring of a message never could.

The base derives from `Exception` rather than `RuntimeError` on purpose. Four
`except RuntimeError` sites in the backend currently swallow domain refusals
alongside genuine faults, and `app/audio/recorder.py:26` records that exact
broad catch as the shape of the JS-107 defect; deriving from `RuntimeError`
would keep those catches silently correct-looking.

Each subclass carries `status_code` and `code` as class variables rather than
having a handler map its type to a status, so adding a refusal with an unusual
status is a class declaration and nothing else. The base's `status_code` of 500
is a sentinel, not a default: a `JustSayError` that answers 500 contradicts
what membership means, and `tests/test_errors.py` fails any subclass that
resolves to it. The base itself cannot be instantiated at all, so the
contradiction has no way to reach the wire — the sentinel is only ever the
inherited value a subclass must override.

Choosing between the three: `ConfigurationError` when the user can fix it in
Settings; `ResourceUnavailableError` when the thing that is wrong lives outside
this process, even if it technically answered; `NotReadyError` only ever about
JustSay's own state machine.

This module imports no web framework and must keep doing so — it has to stay
reachable from every layer. The HTTP translation lives in
`app/core/error_handler.py`, and `tests/test_import_layers.py` holds the line.
"""

from collections.abc import Mapping
from typing import ClassVar


class JustSayError(Exception):
    """A refusal, not a crash.

    `message` is the user-facing sentence and the only part that reaches the
    wire: prose for the person looking at the widget, with no class names,
    tracebacks or file paths in it. `diagnostic` is the raw material a reader
    of the log needs — a provider's reply, a `ctypes` return code, a device id
    — and never leaves the process. `headers` exists so a refusal that already
    answers with `Retry-After` can keep doing so.

    The class is the membership test and never a refusal in its own right:
    instantiating it raises `TypeError`, because the only body it could produce
    is a 500 `internal_error` shaped exactly like a refusal the user is meant
    to read. `except JustSayError` is unaffected — catching is what the base is
    for, raising is not.
    """

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        diagnostic: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if type(self) is JustSayError:
            raise TypeError(
                "JustSayError is the membership test, not a refusal: "
                "raise one of its subclasses instead."
            )
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic
        self.headers = headers


class ConfigurationError(JustSayError):
    """The app cannot do this until something the user controls changes.

    An API key that is absent or rejected, a Cloud/Local mode that forbids the
    request, an output directory that cannot be written, a model size that is
    not installed. The user can fix every one of these in Settings, which is
    what separates it from `ResourceUnavailableError`.
    """

    status_code: ClassVar[int] = 400
    code: ClassVar[str] = "configuration_error"


class ResourceUnavailableError(JustSayError):
    """Something outside the app's control is absent, busy or not answering.

    A held SQLite lock, a missing capture device, a platform without the
    capability at all, an index that has not been built, a helper binary that
    is not on disk, a provider that replied with nothing usable. Nobody can fix
    it from Settings right now; retrying later may work.
    """

    status_code: ClassVar[int] = 503
    code: ClassVar[str] = "resource_unavailable"


class NotReadyError(JustSayError):
    """JustSay's own state forbids this request until the client changes state.

    Not recording, already recording, nothing captured yet, a session that
    belongs to a different caller. Retrying the same request unchanged answers
    the same way, which is why this is a 409 rather than a 503.
    """

    status_code: ClassVar[int] = 409
    code: ClassVar[str] = "not_ready"
