"""Append-only on-disk storage for one capture source of one meeting.

A meeting is captured to disk rather than to memory, so peak RAM stops
depending on how long the call ran (ADR 058). Each source gets a `.pcm` of
mono float32 frames and a sibling `.idx` of fixed-width per-block records, which
is what lets assembly rebuild the arrival timeline without a block in RAM, plus a
`.rate` naming the source's rate for a rebuild at the launch after a crash.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

INDEX_DTYPE = np.dtype([("arrival", "<f8"), ("frames", "<i4")])

SAMPLE_BYTES = np.dtype(np.float32).itemsize

ASSEMBLY_RESERVE_MARGIN_BYTES = 64 * 1024 * 1024


def assembly_reserve_bytes(seconds: float, sample_rate: int, tracks: int) -> int:
    """Free space assembly will need after capturing `seconds` more seconds.

    Both terms — a float32 mix memmap and the int16 WAV read out of it, each
    with `tracks` channels — are derived from `sample_rate` rather than written
    down, so raising the rate cannot silently under-reserve. The margin covers
    the index records.
    """
    per_second = (sample_rate * 4 + sample_rate * 2) * tracks
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


class SpoolFiles:
    """One source's spill files, read back as far as they reached the disk.

    A spool whose writer was killed may end in half a record, or in a record
    whose frames never landed; reading stops at the last block wholly on disk.
    """

    def __init__(self, directory: Path, meeting_id: str, source: str) -> None:
        stem = f"meeting_spill_{meeting_id}_{source}"
        self.meeting_id = meeting_id
        self.samples_path = directory / f"{stem}.pcm"
        self.index_path = directory / f"{stem}.idx"
        self.rate_path = directory / f"{stem}.rate"

    def nominal_rate(self) -> int | None:
        """The rate the writer said this source delivers at, or None if it never said."""
        try:
            return int(self.rate_path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            return None

    def index(self) -> np.ndarray:
        """The arrival record of every block whose frames are all on disk, in order."""
        whole_records = self.index_path.stat().st_size // INDEX_DTYPE.itemsize
        records = np.fromfile(self.index_path, dtype=INDEX_DTYPE, count=whole_records)
        ends = np.cumsum(records["frames"], dtype=np.int64)
        return records[: int(np.searchsorted(ends, self._frames_on_disk(), side="right"))]

    def samples(self) -> np.ndarray:
        """A read-only memmap over every frame on disk."""
        frames = self._frames_on_disk()
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        return np.memmap(self.samples_path, dtype=np.float32, mode="r", shape=(frames,))

    def _frames_on_disk(self) -> int:
        return self.samples_path.stat().st_size // SAMPLE_BYTES

    def discard(self) -> None:
        """Remove every file of this spool, logging rather than raising what stays.

        Each file is tried even when an earlier one stays, and a spool without its
        `.rate` is never rebuilt, so a `.pcm` a scanner still holds open cannot
        come back as a second meeting.
        """
        for path in (self.rate_path, self.index_path, self.samples_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.warning("The meeting spill file %s could not be removed", path)


class MeetingSpool(SpoolFiles):
    """One source's frames on disk, appended in arrival order.

    Arrival order is what makes a segment a contiguous slice of the `.pcm`,
    which is what lets assembly stream.
    """

    def __init__(self, directory: Path, meeting_id: str, source: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        super().__init__(directory, meeting_id, source)
        self._samples_file = open(self.samples_path, "wb")
        self._index_file = open(self.index_path, "wb")
        self._frames = 0

    @property
    def frames(self) -> int:
        """How many frames have been appended so far."""
        return self._frames

    def record_rate(self, rate: int) -> None:
        """Note the rate this source delivers at, which a rebuild after a crash needs."""
        self.rate_path.write_text(str(rate), encoding="ascii")

    def append(self, arrival: float, samples: np.ndarray) -> None:
        """Write one block's frames and the record describing it.

        Each is flushed before the next is written, frames first, so a process
        killed at any instant leaves a spool describing less audio than it holds.
        """
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._samples_file.write(block.tobytes())
        self._samples_file.flush()
        record = np.array([(arrival, block.size)], dtype=INDEX_DTYPE)
        self._index_file.write(record.tobytes())
        self._index_file.flush()
        self._frames += int(block.size)

    def close(self) -> None:
        """Flush both files to disk. Safe to call more than once."""
        for handle in (self._samples_file, self._index_file):
            if not handle.closed:
                handle.flush()
                handle.close()

    def discard(self) -> None:
        """Close both files and remove every file of this spool, never raising."""
        try:
            self.close()
        except OSError:
            pass
        super().discard()
