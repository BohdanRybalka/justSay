from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.app_paths import resolve_app_data_root


class AudioSettings(BaseSettings):
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    # Derived from resolve_app_data_root() -- see docs/adr/012-dev-mode-data-directory-isolation.md.
    # An explicit JUSTSAY_AUDIO_TEMP_DIR env var still overrides this default_factory.
    temp_dir: Path = Field(default_factory=lambda: resolve_app_data_root() / "tmp")

    # Silence-guard thresholds (spec 029 / docs/adr/015-pipeline-level-silence-guard.md).
    # Overridable via JUSTSAY_AUDIO_* env vars for a user with an unusually
    # low-gain mic, without a rebuild. Each of the five below can
    # demonstrably change a verdict -- confirmed at Stage 3 review.
    #
    # silence_frame_dbfs was tuned to -50.0 (not the plan's originally
    # proposed -45.0) against the real sample in train-audio-data/: at -30dB
    # synthetic attenuation, that recording's loudest 30ms frame measured
    # -45.1 dBFS RMS, so -45.0 would have wrongly discarded it.
    #
    # silence_peak_dbfs sits at -45.0, 5 dB ABOVE silence_frame_dbfs -- NOT
    # equal to it. At equal values the peak check is provably dead
    # (rms(frame) <= max|frame| <= global peak, so any frame clearing the
    # frame floor forces the peak to clear an equal peak floor -- confirmed
    # empirically over 400 sampled windows, zero decisive peak checks, Stage
    # 3 review RED-2). At -45.0 it has an exclusive job the frame check
    # structurally cannot do: catching sustained low-crest-factor noise
    # (e.g. a 60 Hz hum whose frames clear the frame floor but whose peak,
    # ~3 dB above its RMS, does not clear a 5 dB-higher peak floor). It
    # cannot misfire on real speech, whose crest factor in the sample is
    # ~25 dB. DO NOT set silence_peak_dbfs <= silence_frame_dbfs.
    silence_peak_dbfs: float = Field(default=-45.0)
    silence_frame_dbfs: float = Field(default=-50.0)
    # Absolute cap on the length-proportional requirement below -- the
    # long-clip case (5 frames, ~150ms).
    silence_min_speech_frames: int = Field(default=5, ge=0)
    # The speech-frame requirement scales with clip length
    # (required = min(silence_min_speech_frames, max(2, ceil(total_frames *
    # silence_min_speech_ratio)))) so a short clip isn't held to an absolute
    # floor it structurally cannot meet (Stage 3 review RED-1: a 200ms clip
    # has only ~7 frames total). Chosen against measurement: 0.15 with a
    # 2-frame floor gives 0/80 false positives on real speech at 0dB/-12dB
    # across 200-1000ms; 0.20/0.25 are worse on quiet clips, 0.10 starts
    # admitting hotkey clicks as speech. The 2-frame floor itself is a
    # module constant in app.audio.analysis, not a field here -- it only
    # matters for sub-100ms-of-speech clips and would be a knob with no
    # real tuning surface.
    silence_min_speech_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    # Fail-open floor: analyze_silence returns None (skip the guard) when
    # fewer than this many ms were actually decoded, measured from decoded
    # samples -- never from sf.info(...).duration, which libsndfile clamps
    # to the physical file size even for a truncated file (Stage 3 review
    # RED-3). Closes the truncated-WAV hole: a header-valid file physically
    # truncated to a handful of bytes decodes to a handful of samples, far
    # under this floor.
    silence_min_analysis_ms: float = Field(default=100.0, ge=0.0)

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_AUDIO_", env_file=".env", extra="ignore")
