"""Meetings a past run left as spill files, rebuilt into recordings at the next launch.

A capture that never reached its WAV — the process crashed or was killed, the
user quit mid-meeting, or the final write failed — leaves its spools on disk
(ADR 058). Nothing here resumes a capture or turns a recording into text: a
rebuilt recording is handed over like any other leftover, for the user to send.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from app.audio.config import AudioSettings
from app.audio.meeting_recorder import (
    MICROPHONE_SOURCE,
    MIX_PREFIX,
    RECORDING_NAME,
    SYSTEM_SOURCE,
    UNFINISHED_SUFFIX,
    _CapturedMeeting,
    write_recording,
)
from app.audio.meeting_spool import SpoolFiles, assembly_reserve_bytes, free_bytes
from app.audio.timeline import TRACK_COUNT

log = logging.getLogger(__name__)

_SPILL_NAME = re.compile(r"meeting_spill_([0-9a-f]{12})_")


def interrupted_meetings(temp_dir: Path) -> list[str]:
    """The ids of the meetings whose spools a past run left behind, oldest first.

    First removes what a cut-off write left: mix files, unfinished WAVs, and the
    spools of a meeting whose WAV is whole. Call it before a capture can start.
    """
    if not temp_dir.is_dir():
        return []
    last_written: dict[str, float] = {}
    for entry in temp_dir.iterdir():
        if _is_unfinished_write(entry.name):
            _remove(entry)
            continue
        found = _SPILL_NAME.match(entry.name)
        if found is not None:
            meeting_id = found[1]
            written = entry.stat().st_mtime
            last_written[meeting_id] = max(last_written.get(meeting_id, written), written)
    pending: list[str] = []
    for meeting_id in sorted(last_written, key=last_written.__getitem__):
        if (temp_dir / f"meeting_{meeting_id}.wav").exists():
            _discard_spill(temp_dir, meeting_id)
        else:
            pending.append(meeting_id)
    return pending


def rebuild_meeting(settings: AudioSettings, meeting_id: str) -> Path | None:
    """Write the meeting's recording from its spools; None when there is nothing to write.

    A set missing a file or a rate, or holding no audio, is removed. Otherwise the
    spools stay for the caller to remove, and a disk too full for the rebuild raises.
    """
    spools = _spools(settings.temp_dir, meeting_id)
    microphone, system = spools
    microphone_rate, system_rate = microphone.nominal_rate(), system.nominal_rate()
    if (
        microphone_rate is None
        or system_rate is None
        or not all(path.exists() for spool in spools for path in _audio_files(spool))
    ):
        log.warning("The spill files of meeting %s are incomplete; removing them", meeting_id)
        _discard_spill(settings.temp_dir, meeting_id)
        return None
    spans = []
    for spool, rate in ((microphone, microphone_rate), (system, system_rate)):
        index = spool.index()
        if index.size:
            last_end = float(index["arrival"][-1]) + int(index["frames"][-1]) / rate
            spans.append((float(index["arrival"][0]), last_end))
    if not spans:
        _discard_spill(settings.temp_dir, meeting_id)
        return None
    recording_start = min(start for start, _ in spans)
    recording_stop = max(stop for _, stop in spans)
    reserve = assembly_reserve_bytes(
        recording_stop - recording_start, settings.sample_rate, TRACK_COUNT
    )
    if free_bytes(settings.temp_dir) < reserve:
        raise OSError(errno.ENOSPC, f"rebuilding this meeting needs {reserve} free bytes")
    cut_off_at = max(path.stat().st_mtime for spool in spools for path in _audio_files(spool))
    recording = write_recording(
        settings,
        _CapturedMeeting(
            microphone_spool=microphone,
            system_spool=system,
            microphone_rate=microphone_rate,
            system_rate=system_rate,
            recording_start=recording_start,
            recording_stop=recording_stop,
            incident=None,
        ),
    )
    os.utime(recording, (cut_off_at, cut_off_at))
    return recording


async def rebuild_interrupted_meetings(
    settings: AudioSettings, meeting_ids: list[str], hand_over: Callable[[list[Path]], None]
) -> None:
    """Rebuild each meeting off the event loop, one at a time, and hand it over.

    A meeting's spools go only once its recording is handed over; one whose
    rebuild or hand-over fails is logged and keeps them for the next launch.
    """
    for meeting_id in meeting_ids:
        try:
            recording = await asyncio.to_thread(rebuild_meeting, settings, meeting_id)
            if recording is None:
                continue
            hand_over([recording])
        except Exception:
            log.error(
                "Rebuilding meeting %s failed; its spill files stay for the next launch",
                meeting_id,
                exc_info=True,
            )
            continue
        log.info("Meeting %s cut off by the last run rebuilt into %s", meeting_id, recording)
        _discard_spill(settings.temp_dir, meeting_id)


def _is_unfinished_write(name: str) -> bool:
    if name.startswith(MIX_PREFIX):
        return True
    finished_name = name.removesuffix(UNFINISHED_SUFFIX)
    return finished_name != name and RECORDING_NAME.fullmatch(finished_name) is not None


def _spools(temp_dir: Path, meeting_id: str) -> tuple[SpoolFiles, SpoolFiles]:
    return (
        SpoolFiles(temp_dir, meeting_id, MICROPHONE_SOURCE),
        SpoolFiles(temp_dir, meeting_id, SYSTEM_SOURCE),
    )


def _audio_files(spool: SpoolFiles) -> tuple[Path, Path]:
    return spool.samples_path, spool.index_path


def _discard_spill(temp_dir: Path, meeting_id: str) -> None:
    for spool in _spools(temp_dir, meeting_id):
        spool.discard()


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("The unfinished meeting file %s could not be removed", path)
