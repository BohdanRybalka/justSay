"""The name a client gives a dictation before it asks for one.

There is one process-wide ``MicrophoneRecorder`` and three surfaces that can
drive it — the widget, the Settings microphone test and ``/pipeline/dictate``
— so "is this recording mine" is the only question that makes a stop, a
discard or an adoption safe, and until spec 119 nothing on the wire could
answer it. The client mints the id, sends it with the request that starts the
capture, and every later request naming it is either the owner or a stranger.
See docs/adr/050-the-client-names-the-recording-before-it-asks-for-one.md.

Pydantic only, never fastapi: docs/style-guide.md §3.2 puts the mapping to an
HTTP status in the router and the raise in the layer below, and
``backend/tests/test_import_layers.py`` fails this module if it grows a web
framework import.
"""

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = "^[0-9a-f]{32}$"

_SESSION_ID_DESCRIPTION = (
    "The client-minted name of one recording: 32 lowercase hexadecimal characters."
)

SESSION_MISMATCH_DETAIL = "That recording belongs to another window"


class SessionRef(BaseModel):
    """A request body naming the recording it means.

    The alphabet is stated once here and enforced by pydantic, so FastAPI
    answers 422 before a handler runs and no hostile value ever reaches the
    recorder. It is deliberately narrow rather than merely bounded: the value
    reaches one string attribute, one JSON response field and one ``==``, and
    ``[0-9a-f]`` carries no separator, no bracket and no newline into any sink
    it might later be given to.

    Its counterpart is ``SESSION_ID_PATTERN`` in ``src/contracts.ts``, pinned
    against this literal by
    ``backend/tests/test_cross_language_contracts.py`` (ADR 045).
    """

    session_id: str = Field(pattern=SESSION_ID_PATTERN, description=_SESSION_ID_DESCRIPTION)


class SessionMismatchError(Exception):
    """A request named a session that does not own the recorder.

    Raised inside the recorder's lock, in the same critical section that
    would have to change for the answer to change, and mapped to ``403`` by
    the router. 403 rather than 409 because the two are the widget's only
    discriminator between "nothing is running" and "something is running and
    it is not yours", and the routers already spend 409 on three other states.
    """
