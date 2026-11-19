"""Wall-clock timeline assembly for a two-source meeting recording.

Pure functions only — no device access, no I/O of its own. Both sources spill
blocks to disk tagged with the `time.monotonic()` reading taken on arrival,
and this module reconciles their two device clocks against that one shared
wall clock (ADR 038), trusting the rate those arrivals show over the rate a
source declares. A meeting file keeps them apart, one channel each.

Everything works in chunks over a memmap and never holds a whole source (ADR 058).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import soxr

MICROPHONE_TRACK = 0
SYSTEM_TRACK = 1
TRACK_COUNT = 2

STANDARD_RATES = (
    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000,
)
_HELD_WINDOWS = 2


@dataclass(frozen=True)
class RateRun:
    """Rows `[start_row, stop_row)` of a spool that its device delivered at `rate`.

    `rate` is a standard rate the arrivals held, or the declared rate where
    nothing could be measured.
    """

    start_row: int
    stop_row: int
    rate: int


@dataclass(frozen=True)
class Segment:
    """A contiguous run of frames at one running rate, with no capture gap inside it.

    Named by its half-open frame range in the source's spool: the spool
    appends in arrival order, so a segment is a slice and never a rebuild.
    """

    start_arrival: float
    end_arrival: float
    start_frame: int
    stop_frame: int
    running_rate: int

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame


def _stall_rows(
    arrivals: np.ndarray, frames: np.ndarray, rates: np.ndarray, gap_tolerance_blocks: float
) -> np.ndarray:
    """Rows arriving more than `gap_tolerance_blocks` earlier blocks after the row before.

    The earlier block lasts its frames at its own entry in `rates`.
    """
    limits = gap_tolerance_blocks * frames[:-1] / rates[:-1]
    return np.flatnonzero(np.diff(arrivals) > limits) + 1


def _row_ranges(breaks: Iterable[int], start: int, stop: int) -> list[tuple[int, int]]:
    edges = [start, *breaks, stop]
    return list(zip(edges, edges[1:]))


def _nearest_rate(measured: float, candidates: Sequence[int]) -> int:
    return min(candidates, key=lambda rate: abs(measured / rate - 1))


def _window_bounds(arrivals: np.ndarray, start: int, stop: int, window_seconds: float) -> list[int]:
    """The first row of each window spanning `window_seconds` of arrivals in `[start, stop)`.

    Window `k` is rows `[bounds[k], bounds[k + 1])`; rows from the last bound on are unmeasured.
    """
    bounds = [start]
    while True:
        following = start + int(
            np.searchsorted(arrivals[start:stop], arrivals[bounds[-1]] + window_seconds)
        )
        if following >= stop:
            return bounds
        bounds.append(following)


def _held_groups(labels: list[int]) -> list[tuple[int, int, int]]:
    """Runs of `_HELD_WINDOWS` or more windows nearest one rate, as `(first, stop, rate)`."""
    groups: list[tuple[int, int, int]] = []
    first = 0
    for rate, members in itertools.groupby(labels):
        count = len(list(members))
        if count >= _HELD_WINDOWS:
            groups.append((first, first + count, rate))
        first += count
    return groups


def _switch_row(
    arrivals: np.ndarray,
    offsets: np.ndarray,
    bounds: list[int],
    before: tuple[int, int, int],
    after: tuple[int, int, int],
) -> int:
    """The first row of `after`'s rate, where the two clocks' lines meet.

    Each line runs through the earliest arrivals of a window wholly at its rate (an
    arrival is only ever late); only the windows touching the switch are searched.
    """

    def intercept(window: int, rate: int) -> float:
        rows = slice(bounds[window], bounds[window + 1])
        return float(np.min(arrivals[rows] - offsets[1:][rows] / rate))

    _, before_stop, before_rate = before
    after_first, _, after_rate = after
    meeting_frame = (
        intercept(after_first + 1, after_rate) - intercept(before_stop - 2, before_rate)
    ) / (1 / before_rate - 1 / after_rate)
    low, high = bounds[before_stop - 1], bounds[after_first + 1]
    return low + int(np.argmin(np.abs(offsets[low:high + 1] - meeting_frame)))


def _piece_runs(
    arrivals: np.ndarray,
    offsets: np.ndarray,
    piece: tuple[int, int],
    candidates: Sequence[int],
    window_seconds: float,
) -> list[tuple[int, int, int]]:
    """A stall-free piece as `(start_row, stop_row, rate)` stretches, empty when no rate held."""
    start, stop = piece
    bounds = _window_bounds(arrivals, start, stop, window_seconds)
    labels = [
        _nearest_rate(
            (offsets[last] - offsets[first]) / (arrivals[last] - arrivals[first]), candidates
        )
        for first, last in zip(bounds, bounds[1:])
    ]
    groups = _held_groups(labels)
    stretches: list[tuple[int, int, int]] = []
    stretch_start = start
    for before, after in zip(groups, groups[1:]):
        if before[2] != after[2]:
            switch = _switch_row(arrivals, offsets, bounds, before, after)
            stretches.append((stretch_start, switch, before[2]))
            stretch_start = switch
    if groups:
        stretches.append((stretch_start, stop, groups[-1][2]))
    return stretches


def find_rate_runs(
    index: np.ndarray,
    nominal_rate: int,
    *,
    gap_tolerance_blocks: float,
    window_seconds: float,
) -> list[RateRun]:
    """Every row of a spool assigned the rate its device really delivered at.

    Measured over windows of `window_seconds`, never across a stall even at the slowest
    standard rate; a rate holds when `_HELD_WINDOWS` windows in a row are nearest to it.
    A piece where none held takes the rate before it, else after, else nominal.
    """
    if index.size == 0:
        return []

    arrivals = index["arrival"]
    frames = index["frames"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(frames)))
    candidates = sorted({*STANDARD_RATES, nominal_rate})
    slowest = np.full(len(index), float(candidates[0]))
    pieces = _row_ranges(
        _stall_rows(arrivals, frames, slowest, gap_tolerance_blocks).tolist(), 0, len(index)
    )
    measured = [
        _piece_runs(arrivals, offsets, piece, candidates, window_seconds) for piece in pieces
    ]

    rate = next((stretches[0][2] for stretches in measured if stretches), nominal_rate)
    runs: list[RateRun] = []
    for (start, stop), stretches in zip(pieces, measured):
        for first, last, stretch_rate in stretches or [(start, stop, rate)]:
            if last <= first:
                continue
            if runs and runs[-1].rate == stretch_rate:
                runs[-1] = RateRun(runs[-1].start_row, last, stretch_rate)
            else:
                runs.append(RateRun(first, last, stretch_rate))
        rate = runs[-1].rate
    return runs


def segment_spool(
    index: np.ndarray, runs: Sequence[RateRun], gap_tolerance_blocks: float
) -> list[Segment]:
    """Split a spool's arrival records wherever capture stalled or its rate changed.

    A stall becomes a hole in the timeline rather than shortening the
    recording: a segment breaks when the gap to the next block exceeds
    `gap_tolerance_blocks` times the earlier block's duration at its run's rate.
    """
    if index.size == 0:
        return []

    arrivals = index["arrival"]
    frames = index["frames"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(frames)))
    rates = np.empty(len(index), dtype=np.float64)
    for run in runs:
        rates[run.start_row:run.stop_row] = run.rate
    breaks = {
        *_stall_rows(arrivals, frames, rates, gap_tolerance_blocks).tolist(),
        *(run.start_row for run in runs[1:]),
    }

    segments: list[Segment] = []
    for first, stop in _row_ranges(sorted(breaks), 0, len(index)):
        last = stop - 1
        rate = int(rates[first])
        segments.append(
            Segment(
                start_arrival=float(arrivals[first]),
                end_arrival=float(arrivals[last]) + int(frames[last]) / rate,
                start_frame=int(offsets[first]),
                stop_frame=int(offsets[stop]),
                running_rate=rate,
            )
        )
    return segments


def segment_effective_rate(segment: Segment, rate_tolerance: float) -> float:
    """The rate the device really ran at over this segment.

    `frames / elapsed_wall_clock`, measured rather than assumed. A measurement
    further than `rate_tolerance` from the segment's running rate is burst
    jitter rather than a crystal, and falls back to that running rate.
    """
    running_rate = float(segment.running_rate)
    elapsed = segment.end_arrival - segment.start_arrival
    if elapsed <= 0:
        return running_rate

    measured = segment.frames / elapsed
    if abs(measured - running_rate) > running_rate * rate_tolerance:
        return running_rate
    return measured


def describe_rate_runs(index: np.ndarray, runs: Sequence[RateRun], recording_start: float) -> str:
    """`runs` for a log line: each rate and the recording second it starts at."""
    starts = (float(index["arrival"][run.start_row]) - recording_start for run in runs)
    return ", ".join(
        f"{run.rate} Hz from {start:.1f} s" for run, start in zip(runs, starts)
    ) or "no blocks"


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
    rate_window_seconds: float,
    chunk_frames: int,
    out: np.ndarray,
) -> list[RateRun]:
    """Add one source's frames to `out` at the offsets their arrivals imply; the rates found.

    `out` spans the whole recording and is summed into, so a silent source
    still contributes its full length. Each segment is written at its own
    absolute offset and truncated at the next one's, so nothing accumulates.
    """
    runs = find_rate_runs(
        index,
        nominal_rate,
        gap_tolerance_blocks=gap_tolerance_blocks,
        window_seconds=rate_window_seconds,
    )
    total_samples = len(out)
    if total_samples == 0:
        return runs

    segments = segment_spool(index, runs, gap_tolerance_blocks)

    for position, segment in enumerate(segments):
        effective_rate = segment_effective_rate(segment, rate_tolerance)
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
    return runs


def normalize_in_place(timeline: np.ndarray, chunk_frames: int) -> None:
    """Scale the timeline down only if it clips, a chunk at a time.

    Division is by the actual peak across every channel, so the sources keep
    their relative loudness and neither is attenuated when the timeline already fits.
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
