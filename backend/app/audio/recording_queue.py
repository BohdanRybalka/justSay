"""What the audio package needs from whoever turns its meeting recordings into text.

``app.audio`` may not import ``app.pipeline``, so ``main.py`` hands it the job queue on
``app.state.jobs``, which meets this protocol.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol


class RecordingQueue(Protocol):
    def receiving(self) -> AbstractContextManager[None]: ...

    def add_meeting(self, recording: Path) -> str: ...

    def kept_recording_names(self) -> frozenset[str]: ...
