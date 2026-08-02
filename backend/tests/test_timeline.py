"""Spec 066: the wall-clock timeline that reconciles two capture clocks.

Pure-function tests with synthetic timestamps — no device, no I/O. This is
where every drift and gap acceptance criterion points. See
docs/adr/038-two-capture-clocks-reconciled-by-measured-rate.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.analysis import to_mono
from app.audio.config import AudioSettings
from app.audio.timeline import (
    CapturedBlock,
    Segment,
    mix_and_normalize,
    place_on_timeline,
    resample_to,
    segment_blocks,
    segment_effective_rate,
)

TARGET_RATE = 16000
GAP_TOLERANCE = 1.5
RATE_TOLERANCE = AudioSettings().meeting_rate_tolerance


def _blocks_at(rate: float, *, block_frames: int, duration: float, start: float = 0.0,
               fill: float = 0.0) -> list[CapturedBlock]:
    """A stream of `block_frames` blocks delivered by a device running at `rate`.

    Arrival timestamps advance by the real time the device took to produce
    each block, so `rate` is genuinely the device's clock and not just a
    label on the data.
    """
    block_duration = block_frames / rate
    count = int(duration / block_duration)
    return [
        CapturedBlock(
            arrival=start + i * block_duration,
            samples=np.full(block_frames, fill, dtype=np.float32),
        )
        for i in range(count)
    ]


def test_drift_is_corrected_and_naive_concatenation_is_not():
    """AC: a 16 020 Hz device over 300 s lands within 50 ms of 300 s.

    This pins the timeline *allocation*, not the drift correction — the two
    marker-position tests below are what pin the correction.

    The anti-vacuity half is the second assertion: the SAME blocks simply
    concatenated are 300.352 s, 7.0x outside the tolerance. Without it a
    do-nothing implementation could pass this test. The naive figure is
    computed from the test's own constants rather than pasted: 300 s at
    16 020 Hz is 4693.35 blocks and a capture delivers whole blocks, so the
    partial block that never arrives is why it is 300.352 and not 300.375.
    """
    block_frames = 1024
    duration = 300.0
    device_rate = 16020.0
    blocks = _blocks_at(device_rate, block_frames=block_frames, duration=duration)

    naive_samples = sum(len(b.samples) for b in blocks)
    naive_seconds = naive_samples / TARGET_RATE
    expected_naive = int(duration * device_rate / block_frames) * block_frames / TARGET_RATE
    assert naive_seconds == pytest.approx(expected_naive, abs=1e-6)
    assert abs(naive_seconds - duration) > 0.050, (
        "the naive path must MISS the tolerance, otherwise the corrected "
        "test below proves nothing"
    )

    corrected = place_on_timeline(
        blocks,
        nominal_rate=TARGET_RATE,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=blocks[-1].arrival + block_frames / 16020.0,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )
    corrected_seconds = len(corrected) / TARGET_RATE

    assert abs(corrected_seconds - naive_seconds) > 0.050
    assert abs(corrected_seconds - duration) <= 0.050


def test_measured_rate_recovers_the_real_device_clock():
    """The correction carries no ppm constant — it measures."""
    blocks = _blocks_at(16020.0, block_frames=1024, duration=60.0)
    segments = segment_blocks(blocks, TARGET_RATE, GAP_TOLERANCE)

    assert len(segments) == 1
    measured = segment_effective_rate(segments[0], TARGET_RATE, RATE_TOLERANCE)
    assert measured == pytest.approx(16020.0, rel=1e-3)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, 0.001, 60.0])
def test_effective_rate_falls_back_to_nominal_when_measurement_is_absurd(elapsed):
    """A degenerate or wildly jittered segment must not resample by 100x.

    The window is +/-`meeting_rate_tolerance` around nominal: 0.001 s for
    16 000 samples measures 16 MHz, 60 s for 16 000 samples measures 267 Hz,
    and a non-positive elapsed measures nothing at all. All three fall back
    to nominal.
    """
    segment = Segment(
        start_arrival=0.0,
        end_arrival=elapsed,
        samples=np.zeros(TARGET_RATE, dtype=np.float32),
    )

    assert segment_effective_rate(segment, TARGET_RATE, RATE_TOLERANCE) == pytest.approx(
        TARGET_RATE
    )


def test_effective_rate_inside_the_window_is_used_as_measured():
    """The window must not swallow the real correction it exists to bound."""
    segment = Segment(
        start_arrival=0.0,
        end_arrival=1.0,
        samples=np.zeros(16020, dtype=np.float32),
    )

    assert segment_effective_rate(segment, TARGET_RATE, RATE_TOLERANCE) == pytest.approx(16020.0)


def test_a_20_second_capture_gap_becomes_silence_not_removed_time():
    """AC: audio after a 20 s stall starts at its true wall-clock offset, and
    the gap itself is silence rather than deleted time."""
    block_frames = 1024
    block_duration = block_frames / TARGET_RATE

    before = _blocks_at(
        TARGET_RATE, block_frames=block_frames, duration=10.0, start=0.0, fill=0.4
    )
    resume_at = before[-1].arrival + block_duration + 20.0
    after = _blocks_at(
        TARGET_RATE, block_frames=block_frames, duration=10.0, start=resume_at, fill=0.4
    )
    stop = after[-1].arrival + block_duration

    timeline = place_on_timeline(
        before + after,
        nominal_rate=TARGET_RATE,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=stop,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    assert len(timeline) / TARGET_RATE == pytest.approx(stop, abs=0.05)

    gap_middle = int((before[-1].arrival + 10.0) * TARGET_RATE)
    assert np.max(np.abs(timeline[gap_middle:gap_middle + TARGET_RATE])) == 0.0

    delivered_before_gap = sum(len(block.samples) for block in before)
    resume_index_for_count = int(resume_at * TARGET_RATE)
    assert np.count_nonzero(timeline[:resume_index_for_count]) == delivered_before_gap

    resume_index = int(resume_at * TARGET_RATE)
    tolerance = int(block_duration * TARGET_RATE)
    audible = np.flatnonzero(np.abs(timeline[resume_index - 5 * tolerance:]) > 1e-6)
    assert audible.size > 0
    first_after_gap = resume_index - 5 * tolerance + int(audible[0])
    assert abs(first_after_gap - resume_index) <= tolerance


def test_a_source_that_delivered_nothing_contributes_a_full_length_silence():
    """A starved loopback endpoint must not shorten the recording."""
    timeline = place_on_timeline(
        [],
        nominal_rate=48000,
        target_rate=TARGET_RATE,
        recording_start=100.0,
        recording_stop=130.0,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    assert len(timeline) == 30 * TARGET_RATE
    assert np.max(np.abs(timeline)) == 0.0


def test_1khz_tone_through_48khz_stereo_survives_as_1khz_mono():
    """AC: the resample path neither aliases the tone nor drops a channel."""
    source_rate = 48000
    block_frames = 1024
    duration = 4.0
    total = int(source_rate * duration)
    t = np.arange(total) / source_rate
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)

    blocks = [
        CapturedBlock(
            arrival=i * block_frames / source_rate,
            samples=to_mono(stereo[i * block_frames:(i + 1) * block_frames]),
        )
        for i in range(total // block_frames)
    ]

    timeline = place_on_timeline(
        blocks,
        nominal_rate=source_rate,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=blocks[-1].arrival + block_frames / source_rate,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    window = timeline[TARGET_RATE // 2: TARGET_RATE // 2 + TARGET_RATE]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    dominant_hz = np.fft.rfftfreq(len(window), 1 / TARGET_RATE)[int(np.argmax(spectrum))]

    assert dominant_hz == pytest.approx(1000.0, abs=10.0)
    assert np.max(np.abs(window)) > 0.3


def test_an_out_of_band_tone_is_filtered_rather_than_folded_into_speech():
    """The reason soxr replaced np.interp (ADR 038).

    A 14 kHz tone is above the 8 kHz Nyquist limit of the 16 kHz output.
    Dropping every third sample — resampling with no anti-alias filter —
    folds it to 16 000 - 14 000 = 2 kHz at nearly full amplitude, straight
    into the speech band the STT provider reads. A band-limited resample
    removes it instead.
    """
    source_rate = 48000
    t = np.arange(source_rate * 2) / source_rate
    tone = (0.9 * np.sin(2 * np.pi * 14000.0 * t)).astype(np.float32)

    band_limited = resample_to(tone, source_rate, TARGET_RATE)
    decimated = tone[::3]

    def _peak_bin_hz(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        return float(np.fft.rfftfreq(len(signal), 1 / TARGET_RATE)[int(np.argmax(spectrum))])

    assert _peak_bin_hz(decimated) == pytest.approx(2000.0, abs=10.0)
    assert np.max(np.abs(decimated)) > 0.8

    steady_state = band_limited[TARGET_RATE // 2:-TARGET_RATE // 2]
    assert np.max(np.abs(steady_state)) < 0.05


def test_mix_stays_within_full_scale_and_keeps_both_sources():
    """AC: a mix that would clip is scaled, not clipped, and both sources
    remain present afterwards."""
    length = TARGET_RATE
    microphone = np.full(length, 0.8, dtype=np.float32)
    system = np.full(length, 0.7, dtype=np.float32)

    mixed = mix_and_normalize(microphone, system)

    assert np.max(np.abs(mixed)) <= 1.0
    assert np.max(np.abs(mixed)) == pytest.approx(1.0, abs=1e-6)

    microphone_only = mix_and_normalize(microphone, np.zeros(length, dtype=np.float32))
    system_only = mix_and_normalize(np.zeros(length, dtype=np.float32), system)
    assert np.max(np.abs(microphone_only)) > 0.0
    assert np.max(np.abs(system_only)) > 0.0
    assert mixed[0] > max(microphone_only[0], system_only[0]) * 0.5


def test_mix_below_full_scale_is_not_attenuated():
    """Scaling only happens when the sum actually clips."""
    microphone = np.full(100, 0.3, dtype=np.float32)
    system = np.full(100, 0.2, dtype=np.float32)

    mixed = mix_and_normalize(microphone, system)

    assert mixed[0] == pytest.approx(0.5)


def test_mix_pads_the_shorter_timeline():
    mixed = mix_and_normalize(
        np.full(10, 0.5, dtype=np.float32), np.full(4, 0.5, dtype=np.float32)
    )

    assert len(mixed) == 10
    assert mixed[0] == pytest.approx(1.0)
    assert mixed[9] == pytest.approx(0.5)


def test_segmentation_splits_only_on_a_real_stall():
    """Ordinary block-to-block jitter must not fragment a stream."""
    block_frames = 1024
    jittered = [
        CapturedBlock(
            arrival=i * (block_frames / TARGET_RATE) * 1.2,
            samples=np.zeros(block_frames, dtype=np.float32),
        )
        for i in range(20)
    ]

    assert len(segment_blocks(jittered, TARGET_RATE, GAP_TOLERANCE)) == 1


def test_segmentation_of_an_empty_stream_is_empty():
    assert segment_blocks([], TARGET_RATE, GAP_TOLERANCE) == []


def test_to_mono_averages_channels_rather_than_taking_the_first():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

    mono = to_mono(stereo)

    assert mono.dtype == np.float32
    assert mono.tolist() == pytest.approx([0.5, 0.5])


def test_blocks_arriving_before_the_recording_start_are_clipped_not_wrapped():
    """A block timestamped before recording_start must be trimmed at the
    front, never written at a negative index (which numpy would silently
    interpret as an offset from the END of the timeline)."""
    block_frames = 1024
    blocks = [
        CapturedBlock(
            arrival=-0.5 + i * block_frames / TARGET_RATE,
            samples=np.full(block_frames, 0.5, dtype=np.float32),
        )
        for i in range(40)
    ]

    timeline = place_on_timeline(
        blocks,
        nominal_rate=TARGET_RATE,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=2.0,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    assert len(timeline) == 2 * TARGET_RATE
    assert np.max(np.abs(timeline[-100:])) > 0.0


def _marker_stream(real_rate: float, *, block_frames: int, duration: float,
                   marker_at: float) -> tuple[list[CapturedBlock], int]:
    """A silent stream from a device running at `real_rate`, with exactly one
    block filled — the block whose arrival span contains `marker_at`.

    Returns the blocks and the index of the marker block, so a test can ask
    where that block's own arrival timestamp says it should land.
    """
    block_duration = block_frames / real_rate
    count = int(duration / block_duration)
    marker_index = int(marker_at / block_duration)
    blocks = [
        CapturedBlock(
            arrival=i * block_duration,
            samples=np.full(
                block_frames, 0.9 if i == marker_index else 0.0, dtype=np.float32
            ),
        )
        for i in range(count)
    ]
    return blocks, marker_index


def _first_audible_second(timeline: np.ndarray) -> float:
    audible = np.flatnonzero(np.abs(timeline) > 0.1)
    assert audible.size > 0, "the marker never made it onto the timeline"
    return int(audible[0]) / TARGET_RATE


def test_a_marker_lands_where_its_own_arrival_says_inside_a_drifted_segment():
    """AC: intra-segment content position — this is what pins the correction.

    Output *length* is fixed by the allocation and is blind to the measured
    rate, so it cannot prove the drift correction runs. What the correction
    controls is where content sits inside that window. Replacing
    `effective_rate` with `float(nominal_rate)` moves this marker 362 ms late,
    72x the tolerance asserted here.
    """
    block_frames = 1024
    device_rate = 16020.0
    duration = 300.0
    marker_at = 290.0
    tolerance = 0.005

    blocks, marker_index = _marker_stream(
        device_rate, block_frames=block_frames, duration=duration, marker_at=marker_at
    )
    marker_arrival = blocks[marker_index].arrival

    assert len(segment_blocks(blocks, TARGET_RATE, GAP_TOLERANCE)) == 1, (
        "a multi-segment stream would let absolute per-segment placement "
        "paper over the drift this test exists to catch"
    )

    uncorrected_displacement = marker_arrival * (device_rate / TARGET_RATE - 1)
    assert tolerance * 20 <= uncorrected_displacement, (
        "the tolerance has been widened to within 20x of the error the "
        "uncorrected path produces, so this test no longer separates them"
    )

    timeline = place_on_timeline(
        blocks,
        nominal_rate=TARGET_RATE,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=blocks[-1].arrival + block_frames / device_rate,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    assert _first_audible_second(timeline) == pytest.approx(marker_arrival, abs=tolerance)


def test_two_capture_clocks_keep_a_common_marker_aligned_with_each_other():
    """AC: cross-source alignment — the observable a user actually hears.

    One marker instant, two devices with different real clocks. The bound is
    block quantisation of the two segment endpoints, written as that
    expression rather than as a rounded literal. Mixing would hide which
    source moved, so the two timelines are inspected separately.
    """
    block_frames = 1024
    duration = 300.0
    marker_at = 290.0
    system_nominal = 48000
    system_real = 48060.0

    def _marker_second(real_rate: float, nominal_rate: int) -> float:
        blocks, _ = _marker_stream(
            real_rate, block_frames=block_frames, duration=duration, marker_at=marker_at
        )
        timeline = place_on_timeline(
            blocks,
            nominal_rate=nominal_rate,
            target_rate=TARGET_RATE,
            recording_start=0.0,
            recording_stop=duration,
            gap_tolerance_blocks=GAP_TOLERANCE,
            rate_tolerance=RATE_TOLERANCE,
        )
        return _first_audible_second(timeline)

    bound = block_frames / TARGET_RATE + block_frames / system_nominal
    uncorrected_skew = duration * (system_real - system_nominal) / system_nominal
    assert uncorrected_skew >= 3 * bound, (
        "the uncorrected path no longer produces a skew this bound can "
        "distinguish, so passing proves nothing"
    )

    skew = abs(
        _marker_second(float(TARGET_RATE), TARGET_RATE)
        - _marker_second(system_real, system_nominal)
    )

    assert skew <= bound


@pytest.mark.parametrize("arrivals", [[0.0, 0.005], [0.0, 0.030], [0.0, 0.032, 0.033]])
def test_burst_jitter_measures_outside_the_window_and_falls_back_to_nominal(arrivals):
    """The measurements the rate window actually has to reject.

    All three land inside the `[0.5x, 2x]` window the code used to carry, so
    that window never rejected anything a real capture produces. Each ratio
    is computed here from its own arrival list, so the test says *why* the
    fallback fires rather than restating the answer.
    """
    nominal_rate = 48000
    block_frames = 1024
    blocks = [
        CapturedBlock(arrival=arrival, samples=np.zeros(block_frames, dtype=np.float32))
        for arrival in arrivals
    ]

    segments = segment_blocks(blocks, nominal_rate, GAP_TOLERANCE)
    assert len(segments) == 1

    elapsed = segments[0].end_arrival - segments[0].start_arrival
    ratio = (len(segments[0].samples) / elapsed) / nominal_rate

    assert abs(ratio - 1.0) > RATE_TOLERANCE, (
        f"a burst measuring {ratio:.6f}x nominal is now inside the "
        f"+/-{RATE_TOLERANCE} window and would be trusted as a clock"
    )
    assert 0.5 <= ratio <= 2.0, (
        "this pattern is inside the old [0.5x, 2x] clamp, which is why that "
        "clamp had to be replaced rather than kept alongside"
    )
    assert segment_effective_rate(segments[0], nominal_rate, RATE_TOLERANCE) == pytest.approx(
        nominal_rate
    )


def test_a_one_block_segment_measures_exactly_nominal_by_construction():
    """The degenerate case the old clamp named and could never have caught.

    `_close_segment` sets `end_arrival = arrival + len(samples)/nominal_rate`,
    so a single block's measured rate is identically nominal — ratio exactly
    1.0, dead centre of any window.
    """
    nominal_rate = 48000
    block_frames = 1024
    segments = segment_blocks(
        [CapturedBlock(arrival=12.5, samples=np.zeros(block_frames, dtype=np.float32))],
        nominal_rate,
        GAP_TOLERANCE,
    )

    assert len(segments) == 1
    elapsed = segments[0].end_arrival - segments[0].start_arrival
    assert (len(segments[0].samples) / elapsed) / nominal_rate == pytest.approx(1.0, abs=1e-12)
    assert segment_effective_rate(segments[0], nominal_rate, RATE_TOLERANCE) == pytest.approx(
        nominal_rate
    )


def test_real_clock_drift_stays_inside_the_rate_window():
    """The window must admit the drift the correction exists for.

    Recomputed from the live `AudioSettings` default, so tightening that
    default until it swallows real drift fails here instead of silently
    disabling the correction.
    """
    drift_ratio = 16020.0 / TARGET_RATE

    assert abs(drift_ratio - 1.0) < RATE_TOLERANCE


def test_a_rejected_segment_is_truncated_at_its_neighbour_rather_than_summed_into_it():
    """AC: no output sample receives audio from two segments of the same source.

    Narrowing the rate window makes the nominal fallback fire more often, and
    the fallback is what causes the overlap: a burst that really took 39 ms is
    laid down as 213 ms of audio at nominal rate, and
    `timeline[...] += resampled[...]` sums it into the next segment instead of
    stopping. The write now stops at the next segment's own offset.
    """
    nominal_rate = 48000
    block_frames = 1024
    fill = 0.6
    block_duration = block_frames / nominal_rate

    burst = [
        CapturedBlock(
            arrival=i * 0.002, samples=np.full(block_frames, fill, dtype=np.float32)
        )
        for i in range(10)
    ]
    resume_at = burst[-1].arrival + GAP_TOLERANCE * block_duration + 0.001
    following = [
        CapturedBlock(
            arrival=resume_at + i * block_duration,
            samples=np.full(block_frames, fill, dtype=np.float32),
        )
        for i in range(20)
    ]
    blocks = burst + following

    assert len(segment_blocks(blocks, nominal_rate, GAP_TOLERANCE)) == 2, (
        "one segment would make this test pass without the truncation"
    )

    timeline = place_on_timeline(
        blocks,
        nominal_rate=nominal_rate,
        target_rate=TARGET_RATE,
        recording_start=0.0,
        recording_stop=following[-1].arrival + block_duration,
        gap_tolerance_blocks=GAP_TOLERANCE,
        rate_tolerance=RATE_TOLERANCE,
    )

    assert np.max(np.abs(timeline)) <= fill * 1.25
