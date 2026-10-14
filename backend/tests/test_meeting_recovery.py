"""A meeting a past run cut off is rebuilt from its spill files at the next launch.

Every spool here is written by the real `MeetingSpool` and closed the way a
dying process's handles are, so the files are exactly what a crash leaves.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.audio.base import write_wav_streaming
from app.audio.config import AudioSettings
from app.audio.meeting_recovery import (
    _discard_spill,
    interrupted_meetings,
    rebuild_interrupted_meetings,
    rebuild_meeting,
)
from app.audio.meeting_spool import MeetingSpool

MICROPHONE_FRAMES = 1600
SYSTEM_RATE = 48000
SYSTEM_FRAMES = 4800
CUT_OFF_AT = 1_700_000_000


@pytest.fixture
def settings(tmp_path: Path) -> AudioSettings:
    return AudioSettings(sample_rate=16000, channels=1, temp_dir=tmp_path / "tmp")


def _left_behind(
    temp_dir: Path,
    meeting_id: str,
    blocks: int = 10,
    written_at: float = CUT_OFF_AT,
) -> list[Path]:
    microphone = MeetingSpool(temp_dir, meeting_id, "microphone")
    system = MeetingSpool(temp_dir, meeting_id, "system")
    microphone.record_rate(16000)
    system.record_rate(SYSTEM_RATE)
    for index in range(blocks):
        microphone.append(
            100.0 + index * MICROPHONE_FRAMES / 16000,
            np.full(MICROPHONE_FRAMES, 0.25, dtype=np.float32),
        )
        system.append(
            100.0 + index * SYSTEM_FRAMES / SYSTEM_RATE,
            np.full(SYSTEM_FRAMES, -0.25, dtype=np.float32),
        )
    microphone.close()
    system.close()
    files = sorted(temp_dir.glob(f"meeting_spill_{meeting_id}_*"))
    for path in files:
        os.utime(path, (written_at, written_at))
    return files


def _channels(recording: Path) -> np.ndarray:
    with wave.open(str(recording)) as wav:
        assert (wav.getnchannels(), wav.getframerate()) == (2, 16000)
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).reshape(-1, 2) / 32767


def test_a_cut_off_meeting_is_rebuilt_into_a_recording_of_both_sides(settings):
    spill = _left_behind(settings.temp_dir, "0123456789ab")

    recording = rebuild_meeting(settings, "0123456789ab")

    assert recording == settings.temp_dir / "meeting_0123456789ab.wav"
    channels = _channels(recording)
    assert len(channels) == 16000
    assert np.median(channels[:, 0]) == pytest.approx(0.25, abs=0.001)
    assert np.median(channels[:, 1]) == pytest.approx(-0.25, abs=0.001)
    assert recording.stat().st_mtime == CUT_OFF_AT, (
        "History names a meeting by its file's time, so the rebuild must carry "
        "the moment the meeting was cut off rather than the moment of the launch"
    )
    assert all(path.exists() for path in spill), (
        "the spools are the only copy until the recording is listed, so the "
        "rebuild leaves them to its caller"
    )


@pytest.mark.asyncio
async def test_the_spools_go_only_once_the_recording_is_handed_over(settings):
    spill = _left_behind(settings.temp_dir, "0123456789ab")
    present_at_hand_over: list[bool] = []

    def hand_over(recordings: list[Path]) -> None:
        present_at_hand_over.extend(path.exists() for path in spill)

    await rebuild_interrupted_meetings(settings, ["0123456789ab"], hand_over)

    assert present_at_hand_over and all(present_at_hand_over)
    assert [path.name for path in spill if path.exists()] == []
    assert (settings.temp_dir / "meeting_0123456789ab.wav").exists()


@pytest.mark.asyncio
async def test_a_hand_over_that_fails_keeps_the_spools_and_goes_on(settings):
    first_spill = _left_behind(settings.temp_dir, "aaaaaaaaaaaa")
    _left_behind(settings.temp_dir, "bbbbbbbbbbbb")
    handed_over: list[Path] = []

    def hand_over(recordings: list[Path]) -> None:
        if recordings[0].name == "meeting_aaaaaaaaaaaa.wav":
            raise FileNotFoundError(recordings[0])
        handed_over.extend(recordings)

    await rebuild_interrupted_meetings(settings, ["aaaaaaaaaaaa", "bbbbbbbbbbbb"], hand_over)

    assert handed_over == [settings.temp_dir / "meeting_bbbbbbbbbbbb.wav"]
    assert all(path.exists() for path in first_spill)


def test_a_spool_a_scanner_still_holds_is_never_rebuilt_into_a_second_meeting(settings):
    """Windows refuses to delete a file another process has open, such as a virus scanner.

    The finished meeting was listed and may be sent and gone by the next launch,
    so a `.pcm` left behind must not come back as a meeting of its own.
    """
    spill = _left_behind(settings.temp_dir, "0123456789ab")
    held = {path for path in spill if path.suffix in (".pcm", ".idx")}
    unlink = Path.unlink

    def refuse_held(path: Path, missing_ok: bool = False) -> None:
        if path in held:
            raise PermissionError(13, "The process cannot access the file", str(path))
        unlink(path, missing_ok=missing_ok)

    recording = rebuild_meeting(settings, "0123456789ab")
    with patch.object(Path, "unlink", refuse_held):
        _discard_spill(settings.temp_dir, "0123456789ab")
    recording.unlink()

    assert interrupted_meetings(settings.temp_dir) == ["0123456789ab"]
    assert rebuild_meeting(settings, "0123456789ab") is None
    assert list(settings.temp_dir.glob("meeting_*.wav")) == []


def test_the_meetings_a_past_run_cut_off_are_found_oldest_first(settings):
    """A finished WAV wins over the spools a crash kept beside it; unfinished writes go."""
    temp_dir = settings.temp_dir
    _left_behind(temp_dir, "aaaaaaaaaaaa", written_at=CUT_OFF_AT + 100)
    _left_behind(temp_dir, "bbbbbbbbbbbb", written_at=CUT_OFF_AT)
    finished_spill = _left_behind(temp_dir, "cccccccccccc")
    finished = temp_dir / "meeting_cccccccccccc.wav"
    finished.write_bytes(b"RIFF")
    unfinished = [
        temp_dir / "meeting_mix_aaaaaaaaaaaa.f32",
        temp_dir / "meeting_aaaaaaaaaaaa.wav.part",
    ]
    unrelated = [
        temp_dir / "rec_0123.wav",
        temp_dir / "meeting_dddddddddddd.wav",
        temp_dir / "download.part",
    ]
    for path in unfinished + unrelated:
        path.write_bytes(b"x")

    assert interrupted_meetings(temp_dir) == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]

    assert [path.name for path in unfinished + finished_spill if path.exists()] == []
    assert finished.exists()
    assert all(path.exists() for path in unrelated)


def test_nothing_is_found_where_there_is_no_scratch_directory(tmp_path):
    assert interrupted_meetings(tmp_path / "never-made") == []


@pytest.mark.parametrize(
    ("blocks", "lost"),
    [
        (10, "meeting_spill_0123456789ab_system.rate"),
        (10, "meeting_spill_0123456789ab_microphone.rate"),
        (10, "meeting_spill_0123456789ab_system.idx"),
        (0, None),
    ],
    ids=["no system rate", "no microphone rate", "a spool file is gone", "no audio"],
)
def test_a_set_that_cannot_become_a_recording_is_removed(settings, blocks, lost):
    spill = _left_behind(settings.temp_dir, "0123456789ab", blocks=blocks)
    if lost is not None:
        (settings.temp_dir / lost).unlink()

    assert rebuild_meeting(settings, "0123456789ab") is None

    assert [path.name for path in spill if path.exists()] == []
    assert list(settings.temp_dir.glob("meeting_*")) == []


def test_a_disk_too_full_for_the_rebuild_keeps_the_spools(settings):
    """Assembly writes through a memory map, and a full disk under one can end the process."""
    spill = _left_behind(settings.temp_dir, "0123456789ab")

    with patch("app.audio.meeting_recovery.free_bytes", return_value=1024), pytest.raises(OSError):
        rebuild_meeting(settings, "0123456789ab")

    assert all(path.exists() for path in spill)
    assert list(settings.temp_dir.glob("meeting_*.wav*")) == []


@pytest.mark.asyncio
async def test_a_rebuild_that_fails_keeps_its_spools_and_the_next_meeting_is_still_rebuilt(
    settings,
):
    first_spill = _left_behind(settings.temp_dir, "aaaaaaaaaaaa")
    _left_behind(settings.temp_dir, "bbbbbbbbbbbb")
    writes: list[Path] = []

    def disk_full_once(path, *args):
        writes.append(path)
        if len(writes) == 1:
            raise OSError("[Errno 28] No space left on device")
        return write_wav_streaming(path, *args)

    handed_over: list[Path] = []
    with patch("app.audio.meeting_recorder.write_wav_streaming", disk_full_once):
        await rebuild_interrupted_meetings(
            settings, ["aaaaaaaaaaaa", "bbbbbbbbbbbb"], handed_over.extend
        )

    assert handed_over == [settings.temp_dir / "meeting_bbbbbbbbbbbb.wav"]
    assert all(path.exists() for path in first_spill)
    assert interrupted_meetings(settings.temp_dir) == ["aaaaaaaaaaaa"]
