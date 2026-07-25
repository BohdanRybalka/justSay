"""Wall-clock timeline assembly for a two-source meeting recording.

Pure functions only — no device access, no I/O. Both capture sources hand
their blocks here tagged with the `time.monotonic()` reading taken when the
block arrived, and this module reconciles the two independent device clocks
against that single shared wall clock. See
docs/adr/038-two-capture-clocks-reconciled-by-measured-rate.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soxr


@dataclass(frozen=True)
class CapturedBlock:
    """One mono float32 block and the wall-clock instant it arrived."""

    arrival: float
    samples: np.ndarray


@dataclass(frozen=True)
class Segment:
    """A contiguous run of blocks with no capture gap inside it."""

    start_arrival: float
    end_arrival: float
    samples: np.ndarray


def segment_blocks(
    blocks: list[CapturedBlock], nominal_rate: int, gap_tolerance_blocks: float
) -> list[Segment]:
    """Split arrival-ordered blocks wherever capture stalled.

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
    if not blocks:
        return []

    segments: list[Segment] = []
    current: list[CapturedBlock] = [blocks[0]]

    for previous, block in zip(blocks, blocks[1:]):
        previous_duration = len(previous.samples) / nominal_rate
        if block.arrival - previous.arrival > gap_tolerance_blocks * previous_duration:
            segments.append(_close_segment(current, nominal_rate))
            current = [block]
        else:
            current.append(block)

    segments.append(_close_segment(current, nominal_rate))
    return segments


def _close_segment(blocks: list[CapturedBlock], nominal_rate: int) -> Segment:
    last = blocks[-1]
    return Segment(
        start_arrival=blocks[0].arrival,
        end_arrival=last.arrival + len(last.samples) / nominal_rate,
        samples=np.concatenate([b.samples for b in blocks]),
    )


def segment_effective_rate(segment: Segment, nominal_rate: int, rate_tolerance: float) -> float:
    """The rate the device really ran at over this segment.

    `len(samples) / elapsed_wall_clock` — measured, never assumed, so no
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

    measured = len(segment.samples) / elapsed
    if abs(measured - nominal_rate) > nominal_rate * rate_tolerance:
        return float(nominal_rate)
    return measured


def place_on_timeline(
    blocks: list[CapturedBlock],
    *,
    nominal_rate: int,
    target_rate: int,
    recording_start: float,
    recording_stop: float,
    gap_tolerance_blocks: float,
    rate_tolerance: float,
) -> np.ndarray:
    """Lay one source's blocks onto a silence-filled timeline at `target_rate`.

    The output always spans the full wall-clock recording, so a source that
    produced nothing contributes silence of the right length instead of
    shortening the result. Each segment is resampled from its own measured
    rate — which removes that segment's mean clock drift exactly — and
    written at the absolute offset its arrival timestamp implies, so
    placement error cannot accumulate from one segment to the next.

    A segment's write stops at the next segment's own placement offset, and
    at the end of the timeline for the last one. While a measured rate is
    used this changes nothing — a segment resampled from `len/elapsed` ends
    exactly at its own `end_arrival`. It matters when `segment_effective_rate`
    falls back to nominal, because a nominal-rate length bears no relation to
    the span the segment really occupied: a packet-bursted segment would
    otherwise be laid across its neighbour's audio and *summed* into it. The
    overrun is dropped rather than mixed, since the samples being discarded
    belong to the segment whose timing was already rejected.
    """
    total_samples = max(int(round((recording_stop - recording_start) * target_rate)), 0)
    timeline = np.zeros(total_samples, dtype=np.float32)
    if total_samples == 0:
        return timeline

    segments = segment_blocks(blocks, nominal_rate, gap_tolerance_blocks)

    for index, segment in enumerate(segments):
        effective_rate = segment_effective_rate(segment, nominal_rate, rate_tolerance)
        resampled = resample_to(segment.samples, effective_rate, target_rate)
        if resampled.size == 0:
            continue

        offset = int(round((segment.start_arrival - recording_start) * target_rate))
        if offset < 0:
            resampled = resampled[-offset:]
            offset = 0
        if offset >= total_samples:
            continue

        limit = total_samples
        if index + 1 < len(segments):
            next_offset = int(
                round((segments[index + 1].start_arrival - recording_start) * target_rate)
            )
            limit = min(limit, next_offset)

        writable = min(len(resampled), limit - offset)
        if writable <= 0:
            continue
        timeline[offset:offset + writable] += resampled[:writable]

    return timeline


def resample_to(samples: np.ndarray, source_rate: float, target_rate: int) -> np.ndarray:
    """Band-limited resample to `target_rate`.

    `soxr`, not linear interpolation. `app.audio.vad` interpolates and its
    own docstring says why that is allowed there and not here: its output
    feeds speech-presence detection and "the file handed to the STT provider
    is never touched by any of this". This output *is* that file, and
    downsampling 48 kHz to 16 kHz without an anti-alias filter folds
    everything above 8 kHz back into the speech band.
    """
    if samples.size == 0:
        return samples.astype(np.float32)
    if abs(source_rate - target_rate) < 1e-9:
        return samples.astype(np.float32)
    return soxr.resample(samples.astype(np.float32), source_rate, target_rate).astype(np.float32)


def to_mono(block: np.ndarray) -> np.ndarray:
    """Downmix an interleaved capture block to mono float32.

    The only work done in a realtime capture callback.
    """
    array = np.asarray(block, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return np.ascontiguousarray(array, dtype=np.float32)


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


def mix_and_normalize(microphone: np.ndarray, system: np.ndarray) -> np.ndarray:
    """Sum the two timelines, scaling down only if the sum clips.

    Division is by the actual peak, so both sources keep their relative
    loudness and neither is attenuated when the sum already fits.
    """
    length = max(len(microphone), len(system))
    mixed = np.zeros(length, dtype=np.float32)
    mixed[: len(microphone)] += microphone
    mixed[: len(system)] += system

    peak = float(np.max(np.abs(mixed))) if length else 0.0
    if peak > 1.0:
        mixed /= peak
    return mixed
