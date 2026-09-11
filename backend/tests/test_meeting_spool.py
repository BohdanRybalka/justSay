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


def test_the_files_carry_the_prefix_the_scratch_cleanup_already_reaps(spool: MeetingSpool):
    """`_SCRATCH_PREFIXES` in `app/preferences/router.py` covers `meeting_`.

    A spool orphaned by a hard crash is reachable by the existing cleanup
    only while its name starts that way, so the name is a contract rather
    than a label.
    """
    assert spool.samples_path.name.startswith("meeting_")
    assert spool.index_path.name.startswith("meeting_")
    assert spool.samples_path.name.endswith("_microphone.pcm")
    assert spool.index_path.name.endswith("_microphone.idx")


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
    target rate, so both terms are that rate -- asserted symbolically against
    the settings rather than against a literal byte count.
    """
    settings = AudioSettings()
    seconds = 3600

    reserve = assembly_reserve_bytes(seconds, settings.sample_rate)

    assert reserve == (
        (settings.sample_rate * 4 + settings.sample_rate * 2) * seconds
        + ASSEMBLY_RESERVE_MARGIN_BYTES
    )
    assert assembly_reserve_bytes(seconds, 48000) > reserve


def test_the_reserve_never_goes_below_its_own_margin():
    """A capture that has just started still needs room to assemble."""
    assert assembly_reserve_bytes(0.0, 16000) == ASSEMBLY_RESERVE_MARGIN_BYTES
    assert assembly_reserve_bytes(-5.0, 16000) == ASSEMBLY_RESERVE_MARGIN_BYTES


def test_free_space_is_read_off_the_volume_the_recording_lands_on(tmp_path: Path):
    assert free_bytes(tmp_path) > 0
