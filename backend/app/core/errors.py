"""The difference between a refusal the user should read and a bug.

A `JustSayError` means the backend declined to do something and can say why in
a sentence a person understands; anything else keeps propagating into a 500.
Each subclass carries its own `status_code` and `code` as class variables, and
the base's 500 is a sentinel a subclass must override. This module imports no
web framework, so every layer can reach it.
"""

from collections.abc import Mapping
from typing import ClassVar


class JustSayError(Exception):
    """A refusal, not a crash. The base is the membership test: only a subclass may be raised.

    `message` is the user-facing sentence and the only part that reaches the wire, free of class
    names, tracebacks and paths. `diagnostic` stays in the log. `headers` carries `Retry-After`.
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

    An absent or rejected API key, a Cloud/Local mode that forbids the request, an unwritable
    output directory, a model size that is not installed — all fixable by the user in Settings.
    """

    status_code: ClassVar[int] = 400
    code: ClassVar[str] = "configuration_error"


class ResourceUnavailableError(JustSayError):
    """Something outside the app's control is absent, busy or not answering.

    A held SQLite lock, a missing capture device, an unbuilt index, a missing helper binary, a bad
    provider reply. Retrying may work; a platform that cannot do it at all answers 501 (ADR 060).
    """

    status_code: ClassVar[int] = 503
    code: ClassVar[str] = "resource_unavailable"


class NotReadyError(JustSayError):
    """JustSay's own state forbids this request until the client changes state.

    Not recording, already recording, nothing captured yet, a session that belongs to a different
    caller. Retrying unchanged answers the same way, which is why this is a 409 and not a 503.
    """

    status_code: ClassVar[int] = 409
    code: ClassVar[str] = "not_ready"
