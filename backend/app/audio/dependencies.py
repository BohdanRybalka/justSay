"""FastAPI wiring for the audio package, and nothing else.

Dependency injection is the only reason this module imports ``fastapi``, and
the four ``Depends()`` accessors below are all of it; ``router.py`` is the
package's other exempt module, and holds the endpoints themselves. Capture
logic, DSP and configuration live in their own modules and stay web-framework
free, so importing one of them does not drag the HTTP stack — or the capture
stack — into the process.
Anything that is not dependency injection belongs elsewhere.
"""

from typing import TypeVar

from fastapi import Request

from app.audio.meeting_recorder import MeetingRecorder
from app.audio.recorder import MicrophoneRecorder

Recorder = TypeVar("Recorder")


def _require(recorder: Recorder | None, attribute: str, dependency: str) -> Recorder:
    """The recorder, or a RuntimeError naming why there is none.

    Two conditions reach this on a real install, not only in tests: the
    lifespan startup did not run at all, and a guarded startup step that
    could not build its recorder and let the backend carry on without it
    (main.py's `_run_optional_step`) -- in which case the WARNING that step
    logged says what failed.
    """
    if recorder is None:
        raise RuntimeError(
            f"app.state.{attribute} is not set — main.py's lifespan startup "
            f"either did not run or could not build it, and the startup log "
            f"says which. In tests, set "
            f"app.dependency_overrides[{dependency}] instead of relying on "
            f"the real recorder."
        )
    return recorder


def get_recorder(request: Request) -> MicrophoneRecorder:
    """FastAPI dependency — the app-lifetime MicrophoneRecorder.

    The instance is created once in main.py's lifespan startup and stored on
    app.state.recorder; this is the Depends() accessor routes use to reach
    it. See docs/adr/005-audio-recorder-di-lifespan.md.
    """
    return _require(get_active_recorder(request), "recorder", "get_recorder")


def get_meeting_recorder(request: Request) -> MeetingRecorder:
    """FastAPI dependency — the app-lifetime MeetingRecorder, when there is one.

    Same shape as get_recorder and built in the same place, but not on the
    same terms: building it is a guarded startup step, because dictation,
    History and Settings all work without it. So an install where that step
    failed serves every other endpoint and raises here, and the meeting
    endpoints answer 500 for the life of the process. See
    docs/adr/005-audio-recorder-di-lifespan.md.
    """
    return _require(
        get_active_meeting_recorder(request), "meeting_recorder", "get_meeting_recorder"
    )


def get_active_recorder(request: Request) -> MicrophoneRecorder | None:
    """The dictation recorder if one exists, otherwise None.

    Used only by the meeting endpoints to refuse starting while Instant
    Prompt holds the microphone. Unlike get_recorder it never raises: a
    missing recorder means nothing is dictating, which is exactly the answer
    the guard needs.
    """
    return getattr(request.app.state, "recorder", None)


def get_active_meeting_recorder(request: Request) -> MeetingRecorder | None:
    """The meeting recorder if one exists, otherwise None.

    The mirror image of get_active_recorder, used by the Instant Prompt
    endpoints to refuse starting while a meeting is being recorded.
    """
    return getattr(request.app.state, "meeting_recorder", None)
