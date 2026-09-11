"""Wall-clock timeline assembly for a two-source meeting recording.

Pure functions only — no device access, no I/O of its own. Both capture
sources spill their blocks to disk tagged with the `time.monotonic()` reading
taken when the block arrived, and this module reconciles the two independent
device clocks against that single shared wall clock. See
docs/adr/038-two-capture-clocks-reconciled-by-measured-rate.md.

Everything here works in chunks over a whole source's frames rather than over
a list of block arrays, because a meeting is captured to disk and assembled
from a memmap — see
docs/adr/058-a-meeting-is-captured-to-disk-not-to-memory.md. Nothing in this
module ever holds a whole source at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import soxr

from app.audio.analysis import to_mono


@dataclass(frozen=True)
class Segment:
    """A contiguous run of frames with no capture gap inside it.

    Named by its half-open frame range in the source's spool rather than by
    its samples: the spool appends in arrival order, so a segment is a slice
    of one file and never has to be rebuilt by concatenation.
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

    A new segment starts when the gap between one block's arrival and the
    next exceeds `gap_tolerance_blocks` times the duration of the earlier
    block itself. A render endpoint that delivers nothing while the machine
    plays silence therefore produces a hole in the timeline rather than
    compressing the recording by the length of the stall.

    Each segment's `end_arrival` is its last block's arrival plus that
    block's own duration at `nominal_rate`, so a segment covers the span its
    audio actually occupies rather than ending at the instant its final block
    started.
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

    `frames / elapsed_wall_clock` — measured, never assumed, so no
    parts-per-million constant for any particular device appears anywhere.

    A measurement further than `rate_tolerance` from nominal is burst jitter
    rather than a crystal, and falls back to nominal. The window is narrow on
    purpose: real clock drift is parts per million, while a short
    packet-bursted segment measures tens of percent off — ratios of 1.62,
    0.83 and 1.18 were reproduced from three arrival patterns at nominal
    48 kHz, all of which the old `[0.5x, 2x]` window admitted. The asymmetry
    of the risk sets the direction: falsely rejecting a long segment throws
    away the whole drift correction, while falsely trusting a short one
    distorts only that segment, so the window errs wide of real drift.
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

    `soxr`, not linear interpolation. `app.audio.vad` interpolates and its
    own docstring says why that is allowed there and not here: its output
    feeds speech-presence detection and "the file handed to the STT provider
    is never touched by any of this". This output *is* that file, and
    downsampling 48 kHz to 16 kHz without an anti-alias filter folds
    everything above 8 kHz back into the speech band.

    `soxr.ResampleStream` rather than `soxr.resample`, because the input is a
    memmap over a file that can be gigabytes long and the output is written
    into another one: the filter state carries across chunk boundaries, so
    the seam between two chunks is not the discontinuity independently
    resampled pieces would leave.
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

    `out` spans the full wall-clock recording and is summed into rather than
    replaced, so the two sources share one buffer and a source that produced
    nothing contributes silence of the right length instead of shortening the
    result. Each segment is resampled from its own measured rate — which
    removes that segment's mean clock drift exactly — and written at the
    absolute offset its arrival timestamp implies, so placement error cannot
    accumulate from one segment to the next.

    A segment's write stops at the next segment's own placement offset, and
    at the end of the timeline for the last one. While a measured rate is
    used this changes nothing — a segment resampled from `frames/elapsed`
    ends exactly at its own `end_arrival`. It matters when
    `segment_effective_rate` falls back to nominal, because a nominal-rate
    length bears no relation to the span the segment really occupied: a
    packet-bursted segment would otherwise be laid across its neighbour's
    audio and *summed* into it. The overrun is dropped rather than mixed,
    since the samples being discarded belong to the segment whose timing was
    already rejected.
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
    loudness and neither is attenuated when the sum already fits. Two passes
    over the buffer rather than one, because the peak is not known until the
    whole mix has been read and the mix is too large to hold in memory.
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


def interleaved_buffer_to_mono(buffer: bytes, channels: int, dtype: str) -> np.ndarray:
    """Read a raw interleaved capture buffer and downmix it to mono float32.

    Both system-audio sources arrive at this same shape from different places —
    a PortAudio callback on Windows, a pipe read from the macOS helper — and
    differ only in dtype spelling. Keeping the deinterleave in one function is
    what stops the two platforms drifting into different channel handling,
    which would be inaudible in tests and obvious in a recording.
    """
    interleaved = np.frombuffer(buffer, dtype=dtype)
    if channels > 1:
        interleaved = interleaved.reshape(-1, channels)
    return to_mono(interleaved)
