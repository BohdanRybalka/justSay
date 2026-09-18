"""The name a client gives a dictation before it asks for one (ADR 050).

One process-wide ``MicrophoneRecorder`` serves several surfaces, so "is this recording mine" is
the question that makes a stop, a discard or an adoption safe. The client mints the id, sends it
with the request that starts the capture, and every later request naming it is the owner or a
stranger.

Pydantic only, never fastapi; a mismatch's status travels on the exception class (ADR 059).
"""

from typing import ClassVar

from pydantic import BaseModel, Field

from app.core.errors import JustSayError

SESSION_ID_PATTERN = "^[0-9a-f]{32}$"

_SESSION_ID_DESCRIPTION = (
    "The client-minted name of one recording: 32 lowercase hexadecimal characters."
)

SESSION_MISMATCH_DETAIL = "That recording belongs to another window"


class SessionRef(BaseModel):
    """A request body naming the recording it means.

    The alphabet is stated once here and enforced by pydantic, so FastAPI answers 422 before a
    handler runs. Its counterpart is ``SESSION_ID_PATTERN`` in ``src/contracts.ts`` (ADR 045).
    """

    session_id: str = Field(pattern=SESSION_ID_PATTERN, description=_SESSION_ID_DESCRIPTION)


class SessionMismatchError(JustSayError):
    """A request named a session that does not own the recorder.

    Answers ``403`` and not the ``409`` its siblings use, because that is the widget's only
    discriminator between "nothing is running" and "something is running and it is not yours".
    """

    status_code: ClassVar[int] = 403
    code: ClassVar[str] = "session_mismatch"
