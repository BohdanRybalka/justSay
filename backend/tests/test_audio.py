import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from app.audio.config import AudioSettings
from app.audio.recorder import MicrophoneRecorder


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


# --- Recording lifecycle ---


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
    await recorder.start()  # second call protected by lock — no-op

    mock_cls.assert_called_once()


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


# --- Audio level ---


@pytest.mark.asyncio
async def test_level_db_updates_during_recording(audio_settings, mock_stream):
    _, _ = mock_stream
    recorder = MicrophoneRecorder(audio_settings)

    assert recorder.level_db == float("-inf")

    await recorder.start()
    _simulate_audio_callback(recorder, num_blocks=3)

    assert recorder.level_db > float("-inf")
    assert recorder.level_db < 0  # dBFS is always negative for non-clipping audio


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
    # Force a known duration by back-dating the start.
    import time
    recorder._start_time = time.monotonic() - 7.5
    await recorder.stop()

    assert recorder.is_recording is False
    # duration_seconds returns 0 after stop (by contract), but last_duration_seconds
    # holds the captured span.
    assert recorder.duration_seconds == 0.0
    assert recorder.last_duration_seconds > 7.0


# --- cleanup() ---


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


def test_cleanup_noop_when_never_started(audio_settings):
    recorder = MicrophoneRecorder(audio_settings)

    recorder.cleanup()  # must not raise

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
    recorder.cleanup()  # second call — must be a genuine no-op

    stream_instance.stop.assert_called_once()
    stream_instance.close.assert_called_once()
    assert recorder.is_recording is False


# --- Config validation ---


def test_config_rejects_negative_sample_rate():
    with pytest.raises(ValidationError):
        AudioSettings(sample_rate=-1, channels=1)


def test_config_rejects_zero_channels():
    with pytest.raises(ValidationError):
        AudioSettings(sample_rate=16000, channels=0)


# --- temp_dir default_factory resolution (spec 020) --------------------------
#
# AudioSettings.temp_dir's default_factory delegates to
# app.core.app_paths.resolve_app_data_root() -- mirrors test_app_paths.py's
# isolation fixture and sys.frozen monkeypatch conventions.


@pytest.fixture
def _isolated_home_and_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("JUSTSAY_DATA_DIR", raising=False)
    monkeypatch.delenv("JUSTSAY_FORCE_DEV_DATA_DIR", raising=False)
    monkeypatch.delenv("JUSTSAY_AUDIO_TEMP_DIR", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    return home


def test_temp_dir_defaults_to_dev_data_root_when_not_frozen(_isolated_home_and_env):
    home = _isolated_home_and_env
    assert AudioSettings().temp_dir == home / ".justsay-dev" / "tmp"


def test_temp_dir_resolves_to_prod_data_root_when_frozen(_isolated_home_and_env, monkeypatch):
    home = _isolated_home_and_env
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    assert AudioSettings().temp_dir == home / ".justsay" / "tmp"


def test_temp_dir_env_override_wins_over_default_factory(
    _isolated_home_and_env, monkeypatch, tmp_path
):
    override = tmp_path / "custom-audio-tmp"
    monkeypatch.setenv("JUSTSAY_AUDIO_TEMP_DIR", str(override))

    assert AudioSettings().temp_dir == override
