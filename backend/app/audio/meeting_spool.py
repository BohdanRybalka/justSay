"""Append-only on-disk storage for one capture source of one meeting.

A meeting is captured to disk rather than to memory, so peak RAM stops
depending on how long the call ran (ADR 058). Each source gets a `.pcm` of
mono float32 frames and a sibling `.idx` of fixed-width per-block records,
which is what lets assembly rebuild the arrival timeline without holding a
block in RAM. Pure storage — no device access, no audio processing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

INDEX_DTYPE = np.dtype([("arrival", "<f8"), ("frames", "<i4")])

ASSEMBLY_RESERVE_MARGIN_BYTES = 64 * 1024 * 1024


def assembly_reserve_bytes(seconds: float, sample_rate: int) -> int:
    """Free space assembly will need after capturing `seconds` more seconds.

    Both terms — a float32 mix memmap and the int16 WAV read out of it — are
    derived from `sample_rate` rather than written down, so raising the rate
    cannot silently under-reserve. The margin covers the index records.
    """
    per_second = sample_rate * 4 + sample_rate * 2
    return int(per_second * max(seconds, 0.0)) + ASSEMBLY_RESERVE_MARGIN_BYTES


def close_memmap(array: np.ndarray) -> None:
    """Release the OS-level mapping behind `array`, if it has one.

    Windows refuses to unlink a mapped file, and a memmap created inside a
    call that raises stays reachable from the traceback — closing explicitly
    is what makes the success and failure paths remove the file alike.
    """
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def free_bytes(directory: Path) -> int:
    """Bytes free on the volume holding `directory`."""
    return shutil.disk_usage(directory).free


class MeetingSpool:
    """One source's frames on disk, appended in arrival order.

    Arrival order is what makes a segment a contiguous slice of the `.pcm`,
    which is what lets assembly stream.
    """

    def __init__(self, directory: Path, meeting_id: str, source: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"meeting_spill_{meeting_id}_{source}"
        self.samples_path = directory / f"{stem}.pcm"
        self.index_path = directory / f"{stem}.idx"
        self._samples_file = open(self.samples_path, "wb")
        self._index_file = open(self.index_path, "wb")
        self._frames = 0

    @property
    def frames(self) -> int:
        """How many frames have been appended so far."""
        return self._frames

    def append(self, arrival: float, samples: np.ndarray) -> None:
        """Write one block's frames and the record describing it.

        The index record is written after the frames, so an interrupted spool
        describes less audio than it holds, never more.
        """
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._samples_file.write(block.tobytes())
        record = np.array([(arrival, block.size)], dtype=INDEX_DTYPE)
        self._index_file.write(record.tobytes())
        self._frames += int(block.size)

    def close(self) -> None:
        """Flush both files to disk. Safe to call more than once."""
        for handle in (self._samples_file, self._index_file):
            if not handle.closed:
                handle.flush()
                handle.close()

    def index(self) -> np.ndarray:
        """The arrival record of every appended block, in order."""
        return np.fromfile(self.index_path, dtype=INDEX_DTYPE)

    def samples(self) -> np.ndarray:
        """A read-only memmap over every appended frame."""
        if self._frames == 0:
            return np.zeros(0, dtype=np.float32)
        return np.memmap(self.samples_path, dtype=np.float32, mode="r")

    def discard(self) -> None:
        """Close both files and remove them, never raising.

        Called from the `finally` that ends every capture, so a spool that has
        already gone must not turn one failure into a different one.
        """
        try:
            self.close()
        except OSError:
            pass
        for path in (self.samples_path, self.index_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
