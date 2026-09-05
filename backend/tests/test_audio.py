import asyncio
import hashlib
import logging
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from app.audio.analysis import SilenceAnalysis, analyze_silence, rms_dbfs
from app.audio.config import AudioSettings
from app.audio.recorder import MicrophoneRecorder
from app.audio.system_source import SystemAudioUnavailableError


@pytest.fixture
def audio_settings(tmp_path):
    return AudioSettings(sample_rate=16000, channels=1, temp_dir=tmp_path)


@pytest.fixture
def mock_stream():
    """Mock sounddevice.InputStream to avoid requiring a real microphone."""
    with patch("app.audio.recorder.sd.InputStream") as mock_cls:
        stream_instance = MagicMock()
        mock_cls.return_value = stream_instance
        yield mock_cls, stream_instance


def _simulate_audio_callback(recorder: MicrophoneRecorder, num_blocks: int = 5):
    """Simulate sounddevice calling the audio callback with fake data."""
    for _ in range(num_blocks):
        fake_audio = np.random.uniform(-0.1, 0.1, (1024, 1)).astype(np.float32)
        recorder._audio_callback(fake_audio, 1024, None, MagicMock())


@pytest.mark.asyncio
async def test_start_recording(audio_settings, mock_stream):
    mock_cls, stream_instance = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()

    assert recorder.is_recording is True
    mock_cls.assert_called_once()
    stream_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_start_idempotent(audio_settings, mock_stream):
    mock_cls, stream_instance = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    await recorder.start()

    mock_cls.assert_called_once()


@pytest.mark.asyncio
async def test_a_failing_stream_constructor_leaves_the_recorder_stopped(audio_settings):
    """A half-started recorder reports `is_recording: true` with nothing to stop.

    `stop()` would then raise "Not recording" forever and `start()` would return
    early, so only a backend restart clears it. MeetingRecorder.start already
    rolls this back; see test_meeting_recorder.py.
    """
    recorder = MicrophoneRecorder(audio_settings)

    with patch("app.audio.recorder.sd.InputStream", side_effect=OSError("no mic")):
        with pytest.raises(OSError):
            await recorder.start()

    assert recorder.is_recording is False
    assert recorder.duration_seconds == 0.0

    with pytest.raises(RuntimeError, match="Not recording"):
        await recorder.stop()


@pytest.mark.asyncio
async def test_a_failing_stream_start_closes_the_half_open_device(audio_settings, mock_stream):
    """The constructor succeeded, so the device stays held until it is closed."""
    _, stream_instance = mock_stream
    stream_instance.start.side_effect = OSError("device busy")
    recorder = MicrophoneRecorder(audio_settings)

    with pytest.raises(OSError):
        await recorder.start()

    assert recorder.is_recording is False
    stream_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_a_recorder_recovers_after_a_failed_start(audio_settings, mock_stream):
    """The point of the rollback: the next start works instead of returning early."""
    recorder = MicrophoneRecorder(audio_settings)

    with patch("app.audio.recorder.sd.InputStream", side_effect=OSError("no mic")):
        with pytest.raises(OSError):
            await recorder.start()

    _, stream_instance = mock_stream
    await recorder.start()

    assert recorder.is_recording is True
    stream_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_stop_returns_wav_file(audio_settings, mock_stream):
    _, stream_instance = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=10)
    audio_path = await recorder.stop()

    assert audio_path.exists()
    assert audio_path.suffix == ".wav"
    assert recorder.is_recording is False

    data, samplerate = sf.read(str(audio_path))
    assert samplerate == 16000
    assert len(data) > 0


@pytest.mark.asyncio
async def test_stop_without_start_raises(audio_settings):
    recorder = MicrophoneRecorder(audio_settings)

    with pytest.raises(RuntimeError, match="Not recording"):
        await recorder.stop()


@pytest.mark.asyncio
async def test_double_stop_raises(audio_settings, mock_stream):
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=3)
    await recorder.stop()

    with pytest.raises(RuntimeError, match="Not recording"):
        await recorder.stop()


@pytest.mark.asyncio
async def test_level_db_updates_during_recording(audio_settings, mock_stream):
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    assert recorder.level_db == float("-inf")

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=3)

    assert recorder.level_db > float("-inf")
    assert recorder.level_db < 0


@pytest.mark.asyncio
async def test_status_not_recording(audio_settings):
    recorder = MicrophoneRecorder(audio_settings)

    assert recorder.is_recording is False
    assert recorder.duration_seconds == 0.0
    assert recorder.level_db == float("-inf")
    assert recorder.last_duration_seconds == 0.0


@pytest.mark.asyncio
async def test_last_duration_persists_after_stop(audio_settings, mock_stream):
    """last_duration_seconds must remain populated after stop() so the pipeline
    can route by audio length without re-reading the WAV."""
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=5)
    import time
    recorder._start_time = time.monotonic() - 7.5
    await recorder.stop()

    assert recorder.is_recording is False
    assert recorder.duration_seconds == 0.0
    assert recorder.last_duration_seconds > 7.0


@pytest.mark.asyncio
async def test_cleanup_stops_and_closes_open_stream(audio_settings, mock_stream):
    _, stream_instance = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=3)
    recorder.cleanup()

    stream_instance.stop.assert_called_once()
    stream_instance.close.assert_called_once()
    assert recorder.is_recording is False


@pytest.mark.asyncio
async def test_cleanup_records_a_stream_close_failure(audio_settings, mock_stream, caplog):
    """`meeting_recorder.py` logs this exact operation and the dictation
    recorder swallowed it, so a device that refuses to close left no evidence
    at all — including on the failed-start rollback, which JS-80 routed through
    this same `cleanup()`.
    """
    _, stream_instance = mock_stream
    stream_instance.stop.side_effect = OSError("PaErrorCode -9988")
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    with caplog.at_level(logging.DEBUG, logger="app.audio.recorder"):
        recorder.cleanup()

    failures = [r for r in caplog.records if r.name == "app.audio.recorder" and r.exc_info]
    assert len(failures) == 1
    assert recorder.is_recording is False


def test_cleanup_noop_when_never_started(audio_settings):
    recorder = MicrophoneRecorder(audio_settings)

    recorder.cleanup()

    assert recorder.is_recording is False


@pytest.mark.asyncio
async def test_cleanup_twice_is_noop_on_second_call(audio_settings, mock_stream):
    """cleanup() must be safe to call repeatedly. The second call on an
    already-cleaned-up recorder must not raise and must not re-invoke
    stream.stop()/close() a second time."""
    _, stream_instance = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=3)
    recorder.cleanup()
    recorder.cleanup()

    stream_instance.stop.assert_called_once()
    stream_instance.close.assert_called_once()
    assert recorder.is_recording is False


_INSTANT_PROMPT_WAV_SHA256 = "1ea56e02eab045bac62debc0a03f936fc08c7a239b9fe7ad8117e0d54fdb11df"


def _seeded_instant_prompt_blocks() -> list[np.ndarray]:
    """The fixed input of the Instant Prompt byte-identity pin.

    Seeded so the block sequence is reproducible on any machine and any
    platform: numpy's `default_rng` is a documented, versioned bit generator,
    so the same seed yields the same float32 samples everywhere.
    """
    rng = np.random.default_rng(66066)
    return [rng.uniform(-0.5, 0.5, (1024, 1)).astype(np.float32) for _ in range(24)]


@pytest.mark.asyncio
async def test_microphone_wav_bytes_unchanged(audio_settings, mock_stream):
    """Spec 066 AC: Instant Prompt is byte-identical.

    The expected digest was captured by running this exact generator against
    `master` (ede926d), before spec 066 wrote a single line of product code.
    Spec 066 adds a second recorder alongside `MicrophoneRecorder` and must
    not move the dictation path by one byte; anything that changes the block
    handling, the int16 conversion or the WAV header turns this red.
    """
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    await recorder.start()
    for block in _seeded_instant_prompt_blocks():
        recorder._audio_callback(block, 1024, None, MagicMock())
    audio_path = await recorder.stop()

    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    assert digest == _INSTANT_PROMPT_WAV_SHA256


@pytest.mark.anyio
async def test_instant_prompt_opens_no_system_audio_source(client, monkeypatch, tmp_path):
    """Spec 066 AC: Instant Prompt opens no system-audio source.

    A meeting recording captures other people's voices. The dictation path
    must not touch that machinery even once — so this makes the factory
    itself fatal and drives POST /audio/start through it.
    """
    from app.audio import get_recorder
    from app.audio import system_source as system_source_module
    from app.main import app as fastapi_app

    def _boom(settings):
        raise AssertionError(
            "the Instant Prompt path called create_system_audio_source()"
        )

    monkeypatch.setattr(system_source_module, "create_system_audio_source", _boom)
    monkeypatch.setattr(
        "app.audio.meeting_recorder.create_system_audio_source", _boom
    )

    settings = AudioSettings(sample_rate=16000, channels=1, temp_dir=tmp_path / "tmp")
    with patch("app.audio.recorder.sd.InputStream") as mock_cls:
        mock_cls.return_value = MagicMock()
        recorder = MicrophoneRecorder(settings)
        fastapi_app.dependency_overrides[get_recorder] = lambda: recorder

        resp = await client.post("/audio/start")

    assert resp.status_code == 200
    assert recorder.is_recording is True
    recorder.cleanup()


def test_config_rejects_negative_sample_rate():
    with pytest.raises(ValidationError):
        AudioSettings(sample_rate=-1, channels=1)


def test_config_rejects_zero_channels():
    with pytest.raises(ValidationError):
        AudioSettings(sample_rate=16000, channels=0)


@pytest.fixture
def isolated_home_and_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("JUSTSAY_DATA_DIR", raising=False)
    monkeypatch.delenv("JUSTSAY_FORCE_DEV_DATA_DIR", raising=False)
    monkeypatch.delenv("JUSTSAY_AUDIO_TEMP_DIR", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    return home


def test_temp_dir_defaults_to_dev_data_root_when_not_frozen(isolated_home_and_env):
    home = isolated_home_and_env
    assert AudioSettings().temp_dir == home / ".justsay-dev" / "tmp"


def test_temp_dir_resolves_to_prod_data_root_when_frozen(isolated_home_and_env, monkeypatch):
    home = isolated_home_and_env
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert AudioSettings().temp_dir == home / ".justsay" / "tmp"


def test_temp_dir_env_override_wins_over_default_factory(
    isolated_home_and_env, monkeypatch, tmp_path
):
    override = tmp_path / "custom-audio-tmp"
    monkeypatch.setenv("JUSTSAY_AUDIO_TEMP_DIR", str(override))

    assert AudioSettings().temp_dir == override


def test_rms_dbfs_matches_manual_formula():
    samples = np.random.uniform(-0.3, 0.3, 2048).astype(np.float32)
    expected = 20 * np.log10(max(np.sqrt(np.mean(samples.astype(np.float64) ** 2)), 1e-10))
    assert rms_dbfs(samples) == pytest.approx(expected)


def test_rms_dbfs_zero_signal_floors_instead_of_raising():
    zero = np.zeros(512, dtype=np.float32)
    assert rms_dbfs(zero) == pytest.approx(-200.0)


@pytest.mark.asyncio
async def test_recorder_level_db_matches_pre_extraction_formula(audio_settings, mock_stream):
    """AC-2: `recorder.level_db` returns the exact same value the old
    inline formula would have computed — not just "some negative number"."""
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)
    await recorder.start()

    fake_audio = np.random.uniform(-0.1, 0.1, (1024, 1)).astype(np.float32)
    recorder._audio_callback(fake_audio, 1024, None, MagicMock())

    rms = np.sqrt(np.mean(fake_audio.astype(np.float64) ** 2))
    expected_dbfs = 20 * np.log10(max(rms, 1e-10))

    assert recorder.level_db == pytest.approx(expected_dbfs)


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def test_analyze_silence_streams_frames_never_reads_whole_file(tmp_path, monkeypatch):
    """AC-3: streaming via soundfile.blocks(), never sf.read() of the whole
    file — memory use must not scale with file length. A regression back to
    sf.read() is caught by making sf.read() itself raise."""
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    def _boom(*a, **kw):
        raise AssertionError("analyze_silence must not call sf.read()")

    monkeypatch.setattr(sf, "read", _boom)

    result = analyze_silence(path, AudioSettings())

    assert result is not None
    assert isinstance(result, SilenceAnalysis)


def test_analyze_silence_returns_none_for_corrupt_file(tmp_path):
    """AC-4 (fail open): a file libsndfile cannot decode returns None, never
    raises, never reports is_silent=True."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is not a real audio file, just garbage bytes" * 4)

    result = analyze_silence(path, AudioSettings())

    assert result is None


def test_analyze_silence_returns_none_for_m4a_container_stub(tmp_path):
    """AC-4 (the load-bearing fail-open case): /pipeline/process-file
    accepts .m4a/.webm, which libsndfile cannot open at all. A guard that
    treated "can't decode" as "silent" would silently break that tab."""
    path = tmp_path / "clip.m4a"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    result = analyze_silence(path, AudioSettings())

    assert result is None


def test_analyze_silence_digital_zero_is_silent(tmp_path):
    """AC-9(a): 3s of true digital silence."""
    path = _write_wav(tmp_path / "zero.wav", np.zeros(16000 * 3))

    result = analyze_silence(path, AudioSettings())

    assert result is not None
    assert result.is_silent is True
    assert result.speech_frame_count == 0


def test_analyze_silence_room_tone_is_silent(tmp_path):
    """AC-9(b): 3s of uniform noise in [-0.0005, 0.0005] (~-66 dBFS peak) —
    a dead mic / room tone, not real speech."""
    rng = np.random.default_rng(1234)
    noise = rng.uniform(-0.0005, 0.0005, 16000 * 3)
    path = _write_wav(tmp_path / "room_tone.wav", noise)

    result = analyze_silence(path, AudioSettings())

    assert result is not None
    assert result.is_silent is True


def test_analyze_silence_verdict_type_is_plain_bool_not_numpy(tmp_path):
    """Regression guard: numpy scalar bools (np.bool_) are not `is`-identical
    to Python's True/False, which would silently break
    `assert result.is_silent is True/False`-style assertions downstream."""
    path = _write_wav(tmp_path / "zero.wav", np.zeros(16000))

    result = analyze_silence(path, AudioSettings())

    assert type(result.is_silent) is bool


def test_analyze_silence_settings_override_flips_verdict(tmp_path):
    """AC-10: thresholds are real AudioSettings fields — an override changes
    the verdict, not just the reported numbers."""
    rng = np.random.default_rng(99)
    noise = rng.uniform(-0.0005, 0.0005, 16000 * 3)
    path = _write_wav(tmp_path / "room_tone.wav", noise)

    assert analyze_silence(path, AudioSettings()).is_silent is True

    permissive = AudioSettings(silence_min_speech_frames=0, silence_peak_dbfs=-100.0)
    assert analyze_silence(path, permissive).is_silent is False


def test_silence_settings_overridable_via_env_var(monkeypatch):
    """AC-10: overridable via JUSTSAY_AUDIO_* env vars without a rebuild."""
    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_PEAK_DBFS", "-100.0")
    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_FRAME_DBFS", "-90.0")
    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_MIN_SPEECH_FRAMES", "1")

    settings = AudioSettings()

    assert settings.silence_peak_dbfs == -100.0
    assert settings.silence_frame_dbfs == -90.0
    assert settings.silence_min_speech_frames == 1


_TRAIN_AUDIO_MP3 = (
    Path(__file__).resolve().parents[2]
    / "train-audio-data"
    / "Record (online-voice-recorder.com).mp3"
)


def _resample_to_16k_mono(data: np.ndarray, orig_sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if orig_sr == 16000:
        return data.astype(np.float32)
    ratio = orig_sr / 16000
    if float(ratio).is_integer():
        factor = int(ratio)
        trimmed = data[: len(data) - (len(data) % factor)]
        return trimmed.reshape(-1, factor).mean(axis=1).astype(np.float32)
    duration = len(data) / orig_sr
    n_target = int(duration * 16000)
    x_orig = np.linspace(0, duration, num=len(data), endpoint=False)
    x_target = np.linspace(0, duration, num=n_target, endpoint=False)
    return np.interp(x_target, x_orig, data).astype(np.float32)


@pytest.fixture(scope="module")
def real_speech_16k_mono() -> np.ndarray | None:
    if not _TRAIN_AUDIO_MP3.exists():
        return None
    data, sr = sf.read(str(_TRAIN_AUDIO_MP3), dtype="float32")
    return _resample_to_16k_mono(data, sr)


@pytest.mark.skipif(
    not _TRAIN_AUDIO_MP3.exists(),
    reason="train-audio-data/ is gitignored and not present in this checkout",
)
@pytest.mark.parametrize("attenuation_db", [0.0, -12.0, -20.0, -30.0])
def test_analyze_silence_does_not_discard_quiet_real_speech(
    tmp_path, real_speech_16k_mono, attenuation_db
):
    """AC-8, the load-bearing false-positive test: the real sample converted
    to 16kHz mono, at 0/-12/-20/-30 dB gain, must never be discarded."""
    attenuated = (real_speech_16k_mono * (10 ** (attenuation_db / 20))).astype(np.float32)
    path = _write_wav(tmp_path / f"speech_{attenuation_db}dB.wav", attenuated)

    result = analyze_silence(path, AudioSettings())

    assert result is not None
    assert result.is_silent is False, (
        f"real speech at {attenuation_db}dB attenuation was wrongly discarded "
        f"(peak={result.peak_dbfs:.1f} dBFS, "
        f"speech_frames={result.speech_frame_count}/{result.total_frame_count})"
    )


def test_rms_dbfs_returns_native_python_float():
    """Item 6 (Stage 3 review nit): match analyze_silence's own casting
    convention — np.log10 of a Python float is still np.float64."""
    samples = np.random.uniform(-0.2, 0.2, 480).astype(np.float32)
    assert type(rms_dbfs(samples)) is float


@pytest.mark.parametrize(
    "total_frame_count,expected_required",
    [(3, 2), (7, 2), (17, 3), (34, 5), (100, 5), (12750, 5)],
)
def test_required_speech_frames_is_proportional_to_clip_length(
    total_frame_count, expected_required
):
    """AC-27: required = min(silence_min_speech_frames, max(2,
    ceil(total_frame_count * silence_min_speech_ratio))) — never the flat
    absolute count that made a 200ms clip (~7 frames) impossible to pass
    at any volume."""
    from app.audio.analysis import _required_speech_frames

    settings = AudioSettings()
    assert _required_speech_frames(total_frame_count, settings) == expected_required


def test_both_speech_requirements_route_through_one_implementation(monkeypatch):
    """Spec 034 AC-11: the anti-drift pin.

    `analysis._required_speech_frames` and `vad._required_speech_hops` used
    to be two structurally identical copies of the same rule, each with its
    own tests — so a change to one would drift from the other silently. They
    are now thin delegations to `analysis.required_speech_units`, and this
    proves it: monkeypatching the shared helper must be visible through BOTH
    wrappers. A "simplification" that inlines either copy back turns this
    red, which is the whole point.

    Each wrapper is also asserted to pass its OWN cap field through, so the
    delegation cannot be proven by a wrapper that ignores its settings."""
    from app.audio import analysis as analysis_module
    from app.audio import vad as vad_module

    seen: list[dict] = []

    def _fake_required_speech_units(total_unit_count, *, cap, ratio):
        seen.append({"total": total_unit_count, "cap": cap, "ratio": ratio})
        return 4242

    monkeypatch.setattr(
        analysis_module, "required_speech_units", _fake_required_speech_units
    )

    settings = AudioSettings()
    assert analysis_module._required_speech_frames(100, settings) == 4242
    assert vad_module._required_speech_hops(100, settings) == 4242

    assert seen == [
        {
            "total": 100,
            "cap": settings.silence_min_speech_frames,
            "ratio": settings.silence_min_speech_ratio,
        },
        {
            "total": 100,
            "cap": settings.silence_vad_min_speech_frames,
            "ratio": settings.silence_min_speech_ratio,
        },
    ]


def test_default_peak_threshold_exceeds_frame_threshold():
    """AC-28's invariant, checked directly: with silence_peak_dbfs <=
    silence_frame_dbfs, rms(frame) <= max|frame| <= global peak makes the
    peak check provably unable to fire before the frame check already has
    (Stage 3 review RED-2) — so the shipped defaults must keep
    silence_peak_dbfs strictly greater than silence_frame_dbfs."""
    settings = AudioSettings()
    assert settings.silence_peak_dbfs > settings.silence_frame_dbfs


def test_analyze_silence_peak_threshold_is_decisive_against_low_crest_factor_hum(tmp_path):
    """AC-28: silence_peak_dbfs must be able to change the verdict on an
    input silence_frame_dbfs/frame-count alone does not resolve.

    Uses a 60Hz SQUARE wave (crest factor ~0dB: peak == RMS), not the sine
    the plan's prose describes -- a sine's crest factor is mathematically
    fixed at ~3.01dB (peak = RMS * sqrt(2)), so hitting peak≈-46.9dBFS while
    RMS still clears the -50dBFS frame floor by a safe margin is not
    simultaneously achievable without sitting within ~1dB of that floor,
    which would make this a flaky test. A square wave at a single -47dBFS
    level (both peak and RMS) demonstrates the identical point -- a
    low-crest-factor signal the frame check alone calls "speech" -- with
    several dB of margin on both sides instead of a fragile 1dB one.
    """
    sr = 16000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr
    amp = 10 ** (-47.0 / 20)
    hum = (amp * np.sign(np.sin(2 * np.pi * 60 * t))).astype(np.float32)
    path = tmp_path / "hum_60hz_square.wav"
    sf.write(str(path), hum, sr, subtype="FLOAT")

    default_result = analyze_silence(path, AudioSettings())
    assert default_result is not None
    assert default_result.is_silent is True, (
        "default silence_peak_dbfs=-45.0 must catch this low-crest-factor hum "
        f"(peak={default_result.peak_dbfs:.1f} dBFS, "
        f"speech_frames={default_result.speech_frame_count}/{default_result.total_frame_count})"
    )
    assert default_result.speech_frame_count >= 5

    overridden = AudioSettings(silence_peak_dbfs=-50.0)
    overridden_result = analyze_silence(path, overridden)
    assert overridden_result is not None
    assert overridden_result.is_silent is False, (
        "with silence_peak_dbfs collapsed onto silence_frame_dbfs, the peak "
        "check is structurally dead and the frame check alone must call "
        "this real (if low-crest-factor) signal 'speech'"
    )


def test_analyze_silence_returns_none_below_min_analysis_ms(tmp_path):
    """AC-29: the guard abstains (None, not a verdict) when it decoded less
    than silence_min_analysis_ms (default 100ms) of audio -- a decode-sanity
    floor, not a short-clip exemption (200ms+ is still judged on content by
    AC-25/26)."""
    audio = np.random.uniform(-0.3, 0.3, int(16000 * 0.050)).astype(np.float32)
    path = tmp_path / "too_short.wav"
    sf.write(str(path), audio, 16000)

    result = analyze_silence(path, AudioSettings())

    assert result is None


def test_analyze_silence_returns_none_for_truncated_wav_with_valid_header(tmp_path):
    """AC-30 (Stage 3 review RED-3): a WAV written in full and then
    physically truncated -- valid header, sf.info() opens it, sf.blocks()
    yields data without raising -- must fail open (None), not report
    is_silent=True. Comparing sf.info(...).frames against frames actually
    read does NOT catch this: libsndfile clamps the declared count to the
    physical file size, so the two already agree on a truncated file. Only
    a decoded-duration floor (AC-29's mechanism) closes this."""
    audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
    path = tmp_path / "truncated.wav"
    sf.write(str(path), audio, 16000)
    with open(path, "r+b") as f:
        f.truncate(60)

    info = sf.info(str(path))
    assert info.frames > 0

    result = analyze_silence(path, AudioSettings())

    assert result is None


@pytest.mark.skipif(
    not _TRAIN_AUDIO_MP3.exists(),
    reason="train-audio-data/ is gitignored and not present in this checkout",
)
def test_analyze_silence_pinned_reviewer_counterexample(tmp_path, real_speech_16k_mono):
    """AC-26: the Stage 3 reviewer's exact counterexample, pinned as a
    regression test. The 200ms window starting at 30.0s of the real sample,
    at 0dB gain, is loud, unattenuated real speech that the iteration-1
    absolute frame-count rule discarded (is_silent=True); the revised
    proportional rule must not."""
    start = int(30.0 * 16000)
    window = real_speech_16k_mono[start:start + int(16000 * 0.200)]
    path = tmp_path / "reviewer_counterexample.wav"
    sf.write(str(path), window, 16000)

    result = analyze_silence(path, AudioSettings())

    assert result is not None
    assert result.total_frame_count == 7
    assert result.speech_frame_count >= 4
    assert result.peak_dbfs > -20.0
    assert result.is_silent is False


@pytest.mark.skipif(
    not _TRAIN_AUDIO_MP3.exists(),
    reason="train-audio-data/ is gitignored and not present in this checkout",
)
@pytest.mark.parametrize("duration_ms", [200, 300, 500, 1000])
@pytest.mark.parametrize("attenuation_db", [0.0, -12.0])
def test_analyze_silence_short_speech_bearing_windows_zero_false_positives(
    tmp_path, real_speech_16k_mono, duration_ms, attenuation_db
):
    """AC-25, the new load-bearing false-positive test. 80 deterministically
    (rng seed 7) sliced windows per cell — accepted only when they
    demonstrably contain a spoken word, not an inter-sentence pause (at 0dB,
    peak_dbfs > -25.0 AND rms_dbfs > -35.0) — must ALL survive the guard at
    every duration from 200ms up, at both 0dB and -12dB gain. Zero false
    positives, not a rate: this is what the iteration-1 absolute frame-count
    rule failed (measured 3/80 at 200ms/0dB, 24-28/80 at 200ms/-12dB,
    11/80 at 300ms/-12dB before this fix)."""
    rng = np.random.default_rng(7)
    audio = real_speech_16k_mono
    win_len = int(16000 * duration_ms / 1000)

    windows: list[np.ndarray] = []
    attempts = 0
    while len(windows) < 80 and attempts < 200_000:
        attempts += 1
        start = int(rng.integers(0, len(audio) - win_len))
        window = audio[start:start + win_len]
        peak_dbfs = 20 * np.log10(max(np.max(np.abs(window)), 1e-10))
        frame_rms_dbfs = rms_dbfs(window)
        if peak_dbfs > -25.0 and frame_rms_dbfs > -35.0:
            windows.append(window)
    assert len(windows) == 80, "could not find 80 speech-bearing windows — sample or seed changed"

    settings = AudioSettings()
    false_positives = []
    for i, window in enumerate(windows):
        attenuated = (window * (10 ** (attenuation_db / 20))).astype(np.float32)
        path = tmp_path / f"w_{duration_ms}ms_{attenuation_db}dB_{i}.wav"
        sf.write(str(path), attenuated, 16000)
        result = analyze_silence(path, settings)
        if result is None or result.is_silent:
            false_positives.append(i)

    assert false_positives == [], (
        f"{len(false_positives)}/80 false positives at {duration_ms}ms/{attenuation_db}dB "
        f"(window indices: {false_positives})"
    )


class _StalledHelper:
    """A spawned macOS helper that never writes its header line."""

    def __init__(self):
        self.stdout = _NeverReadyStream()
        self.terminated = False
        self.returncode = None

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return self.returncode


class _NeverReadyStream:
    def __init__(self):
        self._released = threading.Event()

    def readline(self):
        self._released.wait()
        return b""

    def close(self):
        self._released.set()


def test_a_stalled_macos_helper_fails_instead_of_blocking_forever(monkeypatch):
    """JS-82: a helper that spawns and never answers used to hang the backend.

    `MacOSTapSource.start` is reached from an `async def`, so an unbounded
    `readline()` blocks every endpoint, not just meeting recording.
    """
    from app.audio import macos_tap

    monkeypatch.setattr(macos_tap, "_HEADER_TIMEOUT_SECONDS", 0.2)
    stalled = _StalledHelper()

    started = time.monotonic()
    with pytest.raises(SystemAudioUnavailableError, match="did not answer"):
        macos_tap._read_header(stalled)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    stalled.stdout.close()


def test_a_helper_that_answers_in_time_is_read_normally():
    from app.audio import macos_tap

    class _Prompt:
        stdout = _ImmediateStream(b'{"sample_rate":48000}\n')

    assert macos_tap._read_header(_Prompt()) == b'{"sample_rate":48000}\n'


class _ImmediateStream:
    def __init__(self, line: bytes):
        self._line = line

    def readline(self):
        return self._line


@pytest.mark.asyncio
async def test_dictation_stop_leaves_the_event_loop_free(tmp_path):
    """JS-81: the concatenate and wave write must not run on the loop.

    The write is slowed by a known interval and the floor set proportional to
    it. The original `ticks > 0` was satisfied by a single bare
    `await asyncio.sleep(0)` in front of the inline call, which leaves every
    millisecond of the write on the loop -- JS-97.
    """
    settings = AudioSettings(temp_dir=tmp_path)
    recorder = MicrophoneRecorder(settings)
    recorder._recording = True
    recorder._frames = [np.zeros((settings.sample_rate * 3, settings.channels), dtype=np.float32)]
    recorder._stream = MagicMock()

    blocked_seconds = 0.2
    concatenate_and_write = recorder._concatenate_and_write

    def slow_concatenate_and_write(*args):
        time.sleep(blocked_seconds)
        return concatenate_and_write(*args)

    recorder._concatenate_and_write = slow_concatenate_and_write

    ticks = 0

    async def competitor():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    race = asyncio.ensure_future(competitor())
    written = await recorder.stop()
    race.cancel()

    assert written.exists()
    assert ticks > 100, (
        f"the loop ticked {ticks} times while stop() blocked for "
        f"{blocked_seconds}s -- the concatenate and write are still on the event loop"
    )
