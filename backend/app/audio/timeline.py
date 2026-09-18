"""Wall-clock timeline assembly for a two-source meeting recording.

Pure functions only — no device access, no I/O of its own. Both sources spill
blocks to disk tagged with the `time.monotonic()` reading taken on arrival,
and this module reconciles their two device clocks against that one shared
wall clock (ADR 038).

Everything works in chunks over a memmap and never holds a whole source (ADR 058).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import soxr


@dataclass(frozen=True)
class Segment:
    """A contiguous run of frames with no capture gap inside it.

    Named by its half-open frame range in the source's spool: the spool
    appends in arrival order, so a segment is a slice and never a rebuild.
    """

    start_arrival: float
    end_arrival: float
    start_frame: int
    stop_frame: int

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame


def segment_spool(
    index: np.ndarray, nominal_rate: int, gap_tolerance_blocks: float
) -> list[Segment]:
    """Split a spool's arrival records wherever capture stalled.

    A stall becomes a hole in the timeline rather than shortening the
    recording: a segment breaks when the gap to the next block exceeds
    `gap_tolerance_blocks` times the earlier block's own duration.
    """
    if index.size == 0:
        return []

    arrivals = index["arrival"]
    frames = index["frames"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(frames)))

    def close(first_row: int, stop_row: int) -> Segment:
        last = stop_row - 1
        return Segment(
            start_arrival=float(arrivals[first_row]),
            end_arrival=float(arrivals[last]) + int(frames[last]) / nominal_rate,
            start_frame=int(offsets[first_row]),
            stop_frame=int(offsets[stop_row]),
        )

    segments: list[Segment] = []
    first_row = 0
    for row in range(1, len(index)):
        previous_duration = int(frames[row - 1]) / nominal_rate
        gap = float(arrivals[row]) - float(arrivals[row - 1])
        if gap > gap_tolerance_blocks * previous_duration:
            segments.append(close(first_row, row))
            first_row = row

    segments.append(close(first_row, len(index)))
    return segments


def segment_effective_rate(segment: Segment, nominal_rate: int, rate_tolerance: float) -> float:
    """The rate the device really ran at over this segment.

    `frames / elapsed_wall_clock`, measured rather than assumed. A
    measurement further than `rate_tolerance` from nominal is burst jitter
    rather than a crystal, and falls back to nominal.
    """
    elapsed = segment.end_arrival - segment.start_arrival
    if elapsed <= 0:
        return float(nominal_rate)

    measured = segment.frames / elapsed
    if abs(measured - nominal_rate) > nominal_rate * rate_tolerance:
        return float(nominal_rate)
    return measured


def resample_chunks(
    samples: np.ndarray, source_rate: float, target_rate: int, chunk_frames: int
) -> Iterator[np.ndarray]:
    """Band-limited resample to `target_rate`, a chunk at a time.

    `soxr`, not interpolation: this output is the file the STT provider gets,
    and 48 kHz → 16 kHz with no anti-alias filter folds everything above
    8 kHz into the speech band. Filter state carries across chunk seams.
    """
    source = np.asarray(samples)
    if source.size == 0:
        return
    if abs(source_rate - target_rate) < 1e-9:
        for start in range(0, source.size, chunk_frames):
            yield np.asarray(source[start:start + chunk_frames], dtype=np.float32)
        return

    stream = soxr.ResampleStream(source_rate, target_rate, 1, dtype="float32")
    for start in range(0, source.size, chunk_frames):
        block = np.asarray(source[start:start + chunk_frames], dtype=np.float32)
        resampled = stream.resample_chunk(block, last=start + chunk_frames >= source.size)
        if resampled.size:
            yield np.asarray(resampled, dtype=np.float32)


def place_on_timeline(
    index: np.ndarray,
    samples: np.ndarray,
    *,
    nominal_rate: int,
    target_rate: int,
    recording_start: float,
    gap_tolerance_blocks: float,
    rate_tolerance: float,
    chunk_frames: int,
    out: np.ndarray,
) -> None:
    """Add one source's frames to `out` at the offsets their arrivals imply.

    `out` spans the whole recording and is summed into, so a silent source
    still contributes its full length. Each segment is written at its own
    absolute offset and truncated at the next one's, so nothing accumulates.
    """
    total_samples = len(out)
    if total_samples == 0:
        return

    segments = segment_spool(index, nominal_rate, gap_tolerance_blocks)

    for position, segment in enumerate(segments):
        effective_rate = segment_effective_rate(segment, nominal_rate, rate_tolerance)
        offset = int(round((segment.start_arrival - recording_start) * target_rate))
        if offset >= total_samples:
            continue

        limit = total_samples
        if position + 1 < len(segments):
            next_arrival = segments[position + 1].start_arrival
            limit = min(limit, int(round((next_arrival - recording_start) * target_rate)))

        cursor = offset
        for chunk in resample_chunks(
            samples[segment.start_frame:segment.stop_frame],
            effective_rate,
            target_rate,
            chunk_frames,
        ):
            if cursor >= limit:
                break
            start = max(cursor, 0)
            stop = min(cursor + chunk.size, limit)
            if stop > start:
                out[start:stop] += chunk[start - cursor:stop - cursor]
            cursor += chunk.size


def normalize_in_place(timeline: np.ndarray, chunk_frames: int) -> None:
    """Scale the mixed timeline down only if it clips, a chunk at a time.

    Division is by the actual peak, so both sources keep their relative
    loudness and neither is attenuated when the sum already fits.
    """
    peak = 0.0
    for start in range(0, len(timeline), chunk_frames):
        chunk = timeline[start:start + chunk_frames]
        if chunk.size:
            peak = max(peak, float(np.abs(chunk).max()))

    if peak <= 1.0:
        return
    for start in range(0, len(timeline), chunk_frames):
        timeline[start:start + chunk_frames] /= peak
