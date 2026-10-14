"""Spec 153: the on-disk spool a meeting is captured into.

Pure storage, so every test here is a file on `tmp_path` — no device, no
recorder, no event loop. See
docs/adr/058-a-meeting-is-captured-to-disk-not-to-memory.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio.config import AudioSettings
from app.audio.meeting_spool import (
    ASSEMBLY_RESERVE_MARGIN_BYTES,
    INDEX_DTYPE,
    MeetingSpool,
    SpoolFiles,
    assembly_reserve_bytes,
    free_bytes,
)


@pytest.fixture
def spool(tmp_path: Path) -> MeetingSpool:
    return MeetingSpool(tmp_path / "spill", "abc123", "microphone")


def test_appended_blocks_read_back_as_one_contiguous_run(spool: MeetingSpool):
    """Assembly slices a segment out of the `.pcm` instead of concatenating it.

    That is only sound if the frames land in the file in the order they were
    appended and with nothing between them, which is what this drives.
    """
    blocks = [np.full(4, index, dtype=np.float32) for index in range(5)]
    for index, block in enumerate(blocks):
        spool.append(10.0 + index, block)
    spool.close()

    assert spool.frames == 20
    assert np.array_equal(np.asarray(spool.samples()), np.concatenate(blocks))


def test_the_index_carries_one_record_per_block_in_arrival_order(spool: MeetingSpool):
    spool.append(1.5, np.zeros(1024, dtype=np.float32))
    spool.append(2.5, np.zeros(512, dtype=np.float32))
    spool.close()

    index = spool.index()

    assert index.dtype == INDEX_DTYPE
    assert list(index["arrival"]) == [1.5, 2.5]
    assert list(index["frames"]) == [1024, 512]


def test_a_spool_nothing_was_appended_to_reads_back_as_nothing(spool: MeetingSpool):
    """The empty capture must not need a special case at the caller.

    `np.memmap` refuses a zero-length file, so a spool that never saw a block
    answers with an empty array rather than raising into the write path.
    """
    spool.close()

    assert spool.frames == 0
    assert spool.samples().size == 0
    assert spool.index().size == 0


def test_the_files_carry_the_names_the_launch_scan_reads(spool: MeetingSpool):
    """`app/audio/meeting_recovery.py` finds a cut-off meeting by these names.

    A spool orphaned by a crash is rebuilt only while its name says which
    meeting and which source it holds, so the name is a contract.
    """
    spool.record_rate(48000)

    assert spool.samples_path.name == "meeting_spill_abc123_microphone.pcm"
    assert spool.index_path.name == "meeting_spill_abc123_microphone.idx"
    assert spool.rate_path.name == "meeting_spill_abc123_microphone.rate"
    assert SpoolFiles(spool.samples_path.parent, "abc123", "microphone").nominal_rate() == 48000


def test_every_appended_block_is_on_disk_before_the_spool_is_closed(spool: MeetingSpool):
    """A killed process runs no `close()`, so what it leaves is what was flushed."""
    spool.append(1.0, np.full(1024, 0.5, dtype=np.float32))
    spool.append(1.064, np.full(1024, 0.25, dtype=np.float32))

    left = SpoolFiles(spool.samples_path.parent, "abc123", "microphone")

    assert list(left.index()["frames"]) == [1024, 1024]
    assert np.array_equal(
        np.asarray(left.samples()),
        np.concatenate([np.full(1024, 0.5), np.full(1024, 0.25)]).astype(np.float32),
    )
    spool.close()


def test_a_cut_off_spool_reads_only_the_blocks_wholly_on_disk(spool: MeetingSpool):
    """A kill can land between a block's frames and its record, or inside either."""
    for index in range(3):
        spool.append(float(index), np.zeros(100, dtype=np.float32))
    spool.close()
    with open(spool.samples_path, "r+b") as samples:
        samples.truncate(250 * 4 + 2)
    with open(spool.index_path, "ab") as index:
        index.write(bytes(5))

    left = SpoolFiles(spool.samples_path.parent, "abc123", "microphone")

    assert list(left.index()["arrival"]) == [0.0, 1.0]
    assert left.samples().size == 250


def test_a_spool_with_no_recorded_rate_says_so(spool: MeetingSpool):
    assert SpoolFiles(spool.samples_path.parent, "abc123", "microphone").nominal_rate() is None


def test_appending_to_a_closed_spool_raises_rather_than_losing_the_block(spool: MeetingSpool):
    """The spill thread turns this into a `storage_failed` incident.

    Swallowing it here would leave the recorder believing audio it never
    wrote was captured.
    """
    spool.close()

    with pytest.raises(ValueError):
        spool.append(1.0, np.zeros(8, dtype=np.float32))


def test_a_discarded_spool_leaves_nothing_behind(spool: MeetingSpool):
    spool.append(1.0, np.zeros(8, dtype=np.float32))
    spool.record_rate(16000)

    spool.discard()

    assert not spool.samples_path.exists()
    assert not spool.index_path.exists()


def test_discarding_twice_is_not_an_error(spool: MeetingSpool):
    """It runs from the `finally` of a write that may itself have failed."""
    spool.discard()
    spool.discard()

    assert not spool.samples_path.exists()


def test_the_assembly_reserve_is_derived_from_the_sample_rate_not_written_down():
    """AC: raising `sample_rate` alone cannot silently under-reserve.

    The mix file is float32 at the target rate and the WAV is int16 at the
    target rate, each with a channel per track, so every term is that rate --
    asserted symbolically against the settings rather than a literal byte count.
    """
    settings = AudioSettings()
    seconds = 3600

    reserve = assembly_reserve_bytes(seconds, settings.sample_rate, 2)

    assert reserve == (
        (settings.sample_rate * 4 + settings.sample_rate * 2) * 2 * seconds
        + ASSEMBLY_RESERVE_MARGIN_BYTES
    )
    assert assembly_reserve_bytes(seconds, 48000, 2) > reserve
    assert assembly_reserve_bytes(seconds, settings.sample_rate, 1) < reserve


def test_the_reserve_never_goes_below_its_own_margin():
    """A capture that has just started still needs room to assemble."""
    assert assembly_reserve_bytes(0.0, 16000, 2) == ASSEMBLY_RESERVE_MARGIN_BYTES
    assert assembly_reserve_bytes(-5.0, 16000, 2) == ASSEMBLY_RESERVE_MARGIN_BYTES


def test_free_space_is_read_off_the_volume_the_recording_lands_on(tmp_path: Path):
    assert free_bytes(tmp_path) > 0
