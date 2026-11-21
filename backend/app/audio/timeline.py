"""Wall-clock timeline assembly for a two-source meeting recording.

Pure functions only — no device access, no I/O of its own. Both sources spill
blocks to disk tagged with the `time.monotonic()` reading taken on arrival,
and this module reconciles their two device clocks against that one shared
wall clock (ADR 038). A meeting file keeps them apart, one channel each. The
rate a source's device delivered at is also measured, for the log; placement never reads it.

Everything works in chunks over a memmap and never holds a whole source (ADR 058).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
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
_SPANS_PER_WINDOW = 4
_MATCH_TOLERANCE = 0.03
_FAR_RATIO = 1.3


@dataclass(frozen=True)
class RateRun:
    """From `start_row` until the next run's, a spool's device delivered at `rate`.

    `rate` is the candidate the arrivals held, the nominal rate or a standard one,
    or the nominal rate when none held anywhere in the spool.
    """

    start_row: int
    rate: int


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


def _spool_columns(index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrivals, frames per block, and each block's first frame followed by the total.

    Arrivals come back as a contiguous copy of the record array's field.
    """
    frames = index["frames"].astype(np.int64)
    arrivals = np.ascontiguousarray(index["arrival"])
    return arrivals, frames, np.concatenate(([0], np.cumsum(frames)))


def _stall_rows(
    arrivals: np.ndarray, frames: np.ndarray, rate: float, gap_tolerance_blocks: float
) -> np.ndarray:
    """Rows arriving more than `gap_tolerance_blocks` blocks at `rate` after the row before."""
    limits = gap_tolerance_blocks * (frames[:-1] / rate)
    return np.flatnonzero(np.diff(arrivals) > limits) + 1


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

    arrivals, frames, offsets = _spool_columns(index)
    stalls = _stall_rows(arrivals, frames, float(nominal_rate), gap_tolerance_blocks)
    edges = [0, *stalls.tolist(), len(index)]
    return [
        Segment(
            start_arrival=float(arrivals[first]),
            end_arrival=float(arrivals[stop - 1]) + int(frames[stop - 1]) / nominal_rate,
            start_frame=int(offsets[first]),
            stop_frame=int(offsets[stop]),
        )
        for first, stop in zip(edges, edges[1:])
    ]


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


def _window_bounds(arrivals: np.ndarray, window_seconds: float) -> list[int]:
    """The first row of each window of `window_seconds` of arrivals, then the row count.

    Window `k` is rows `[bounds[k], bounds[k + 1])`.
    """
    bounds = [0]
    while bounds[-1] < len(arrivals):
        bounds.append(int(np.searchsorted(arrivals, arrivals[bounds[-1]] + window_seconds)))
    return bounds


def _window_labels(
    columns: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: list[int],
    candidates: Sequence[int],
    gap_tolerance_blocks: float,
    window_seconds: float,
) -> list[int | None]:
    """Each window's rate: the first of `candidates` its own stall-free spans measure.

    A row spans to the row a `_SPANS_PER_WINDOW`th of a window on, and counts for a candidate
    only when no gap inside its span is a stall at that rate; the spans' median must lie
    within `_MATCH_TOLERANCE` of the candidate. None when no candidate agrees.
    """
    arrivals, frames, offsets = columns
    count = len(arrivals)
    rows = np.arange(count)
    ends = np.searchsorted(arrivals, arrivals + window_seconds / _SPANS_PER_WINDOW)
    spanned = ends < count
    span_rates = np.full(count, np.nan)
    block_ends = offsets[1:]
    span_rates[spanned] = (block_ends[ends[spanned]] - block_ends[spanned]) / (
        arrivals[ends[spanned]] - arrivals[spanned]
    )
    stall_free = {}
    for rate in candidates:
        stalls = _stall_rows(arrivals, frames, float(rate), gap_tolerance_blocks)
        piece_stops = np.append(stalls, count)[np.searchsorted(stalls, rows, side="right")]
        stall_free[rate] = spanned & (ends < piece_stops)

    labels: list[int | None] = []
    for first, stop in zip(bounds, bounds[1:]):
        agreed = None
        for rate in candidates:
            usable = stall_free[rate][first:stop]
            if not usable.any():
                continue
            measured = float(np.median(span_rates[first:stop][usable]))
            if abs(measured / rate - 1) <= _MATCH_TOLERANCE:
                agreed = rate
                break
        labels.append(agreed)
    return labels


def _held_groups(labels: list[int | None]) -> list[tuple[int, int, int]]:
    """Runs of `_HELD_WINDOWS`+ labelled windows on one rate, as `(first, last, rate)`.

    Unlabelled windows are passed over rather than ending a run.
    """
    labelled = [(window, rate) for window, rate in enumerate(labels) if rate is not None]
    groups: list[tuple[int, int, int]] = []
    for rate, members in itertools.groupby(labelled, key=lambda item: item[1]):
        windows = [window for window, _ in members]
        if len(windows) >= _HELD_WINDOWS:
            groups.append((windows[0], windows[-1], rate))
    return groups


def _switch_row(
    arrivals: np.ndarray,
    frames: np.ndarray,
    search: tuple[int, int],
    rates: tuple[int, int],
) -> int:
    """The row in `[low, high]` from which blocks last as long as the second rate says.

    The least summed misfit of each block's duration decides, so a stall moves the switch
    by at most its own block; a tie goes to the earlier row.
    """
    low, high = search
    before_rate, after_rate = rates
    rows = np.arange(max(low, 1), high)
    durations = arrivals[rows] - arrivals[rows - 1]
    misfit_before = np.abs(durations - frames[rows] / before_rate)
    misfit_after = np.abs(durations - frames[rows] / after_rate)
    cost = np.concatenate(([0.0], np.cumsum(misfit_before))) + np.concatenate(
        (np.cumsum(misfit_after[::-1])[::-1], [0.0])
    )
    return int(rows[0]) + int(np.argmin(cost))


def find_rate_runs(
    index: np.ndarray,
    nominal_rate: int,
    *,
    gap_tolerance_blocks: float,
    window_seconds: float,
) -> list[RateRun]:
    """Every row of a spool assigned the rate its device delivered at, as far as arrivals tell.

    Candidates are the nominal rate, then standard rates `_FAR_RATIO` or more from it, since
    lost time inside a span reads as a slightly slower clock. A rate holds when `_HELD_WINDOWS`
    labelled windows in a row carry it; a switch lands where block durations change. Rows
    before the first held rate take it, and with none held every row takes the nominal rate.
    """
    if index.size == 0:
        return []

    columns = _spool_columns(index)
    arrivals, frames, _ = columns
    candidates = [
        nominal_rate,
        *(
            rate
            for rate in STANDARD_RATES
            if max(rate / nominal_rate, nominal_rate / rate) >= _FAR_RATIO
        ),
    ]
    bounds = _window_bounds(arrivals, window_seconds)
    groups = _held_groups(
        _window_labels(columns, bounds, candidates, gap_tolerance_blocks, window_seconds)
    )
    if not groups:
        return [RateRun(0, nominal_rate)]

    runs = [RateRun(0, groups[0][2])]
    for before, after in zip(groups, groups[1:]):
        if before[2] != after[2]:
            search = (bounds[before[1]], bounds[after[0] + 1])
            switch = _switch_row(arrivals, frames, search, (before[2], after[2]))
            runs.append(RateRun(switch, after[2]))
    return runs


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
