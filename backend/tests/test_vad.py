"""Tests for the TEN VAD neural silence gate (spec 033 / ADR 019).

Two tiers, deliberately separated:

  - Tier 1 (always runs): resolution chain, fail-open rungs, verdict rule,
    streaming/early-exit, native-type pins. These use monkeypatched or
    stubbed library handles, so they are deterministic on ANY checkout,
    with or without the vendored binary.
  - Tier 2 ([DLL], skipif-guarded): behavioural claims that only mean
    something against the real neural model — the real-speech sweep, the
    averted-false-positive case, loud non-speech, latency. These skip
    loudly on a checkout that never ran `backend/scripts/fetch_ten_vad.py`,
    exactly like the existing train-audio-data/ skips.

A silently-skipped tier 2 is the failure mode to watch for: if these skip
on a machine where the DLL IS present, the neural layer is untested and the
skip reason will say why.
"""

import ctypes
import math
import statistics
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.audio import vad as vad_module
from app.audio.analysis import analyze_silence, rms_dbfs
from app.audio.config import AudioSettings
from app.audio.vad import (
    VadAnalysis,
    _required_speech_hops,
    analyze_vad,
    resolve_ten_vad_lib,
)
from app.pipeline import service as service_module
from app.pipeline.service import process_audio
from app.stt.base import TranscriptionResult

_TEN_VAD_LIB = vad_module.resolve_ten_vad_lib()
_HAS_DLL = _TEN_VAD_LIB is not None
_requires_dll = pytest.mark.skipif(
    not _HAS_DLL,
    reason=(
        "TEN VAD binary not vendored in this checkout — run "
        "backend/scripts/fetch_ten_vad.py to enable the [DLL] behavioural tests"
    ),
)

_TRAIN_AUDIO_MP3 = (
    Path(__file__).resolve().parents[2]
    / "train-audio-data"
    / "Record (online-voice-recorder.com).mp3"
)
_requires_sample = pytest.mark.skipif(
    not _TRAIN_AUDIO_MP3.exists(),
    reason="train-audio-data/ is gitignored and not present in this checkout",
)


@pytest.fixture(autouse=True)
def _reset_library_cache():
    """The library handle (and a failed load) is cached per process — drop it
    around every test so a monkeypatched resolver actually takes effect and
    no test leaks a cached handle into the next."""
    vad_module._reset_library_cache()
    yield
    vad_module._reset_library_cache()


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def _speech_16k_mono() -> np.ndarray:
    """The real sample, decoded to 16 kHz mono (mirrors test_audio.py's helper)."""
    data, sr = sf.read(str(_TRAIN_AUDIO_MP3), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    factor = int(sr / 16000)
    trimmed = data[: len(data) - (len(data) % factor)]
    return trimmed.reshape(-1, factor).mean(axis=1).astype(np.float32)


def _speech_bearing_windows_with_starts(audio: np.ndarray, duration_ms: int, count: int = 80):
    """Spec 029's AC-25 window selection, reused VERBATIM (deterministic
    rng(7), speech-bearing windows only) so this sweep is directly
    comparable to the energy guard's own measured numbers.

    Returns (start_sample, window) pairs — the start is what lets AC 11(c)
    pin the IDENTITY of its one tolerated net-new discard, not just its count."""
    rng = np.random.default_rng(7)
    win_len = int(16000 * duration_ms / 1000)
    windows: list[tuple[int, np.ndarray]] = []
    attempts = 0
    while len(windows) < count and attempts < 200_000:
        attempts += 1
        start = int(rng.integers(0, len(audio) - win_len))
        window = audio[start:start + win_len]
        peak_dbfs = 20 * np.log10(max(np.max(np.abs(window)), 1e-10))
        if peak_dbfs > -25.0 and rms_dbfs(window) > -35.0:
            windows.append((start, window))
    assert len(windows) == count, (
        "could not find enough speech-bearing windows — sample or seed changed"
    )
    return windows


# --- AC 2: library resolution chain ---------------------------------------


def test_resolve_prefers_env_override(tmp_path, monkeypatch):
    """AC-2: JUSTSAY_TEN_VAD_LIB wins over every other source."""
    override = tmp_path / "custom_ten_vad.dll"
    override.write_bytes(b"stub")
    monkeypatch.setenv("JUSTSAY_TEN_VAD_LIB", str(override))

    assert resolve_ten_vad_lib() == override


def test_resolve_falls_through_when_env_override_does_not_exist(tmp_path, monkeypatch):
    """AC-2: a stale env var pointing at a deleted file must DEGRADE to the
    next source, never hard-fail the dictation. Here nothing else exists
    either, so the whole chain yields None."""
    monkeypatch.setenv("JUSTSAY_TEN_VAD_LIB", str(tmp_path / "gone.dll"))
    monkeypatch.setattr(vad_module, "__file__", str(tmp_path / "app" / "audio" / "vad.py"))

    assert resolve_ten_vad_lib() is None


def test_resolve_returns_none_when_nothing_found(tmp_path, monkeypatch):
    """AC-2: no override, not frozen, no vendor dir -> None (the normal
    outcome on every non-Windows platform and every un-fetched checkout)."""
    monkeypatch.delenv("JUSTSAY_TEN_VAD_LIB", raising=False)
    monkeypatch.setattr(vad_module, "__file__", str(tmp_path / "app" / "audio" / "vad.py"))
    monkeypatch.setattr(vad_module.sys, "frozen", False, raising=False)

    assert resolve_ten_vad_lib() is None


def test_resolve_uses_frozen_bundle_path(tmp_path, monkeypatch):
    """AC-2: inside the PyInstaller sidecar the DLL lives at
    sys._MEIPASS/ten_vad/<lib>, which is where build_sidecar.spec puts it."""
    monkeypatch.delenv("JUSTSAY_TEN_VAD_LIB", raising=False)
    bundled = tmp_path / "ten_vad" / vad_module._platform_lib_name()
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"stub")
    monkeypatch.setattr(vad_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(vad_module.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert resolve_ten_vad_lib() == bundled


# --- AC 6: verdict rule ----------------------------------------------------


@pytest.mark.parametrize(
    "total_hops,expected_required",
    [(1, 2), (12, 2), (13, 2), (20, 3), (34, 5), (200, 5), (10_000, 5)],
)
def test_required_speech_hops_is_proportional_to_clip_length(total_hops, expected_required):
    """AC-6: required = min(silence_vad_min_speech_frames, max(2,
    ceil(total_hops * silence_min_speech_ratio))) — the floor-of-2 +
    reused-ratio convention from analysis.py, so a 200ms clip (~12 hops) is
    not held to an absolute bar it structurally cannot clear (029 RED-1)."""
    assert _required_speech_hops(total_hops, AudioSettings()) == expected_required


def test_required_speech_hops_reuses_shipped_ratio_not_a_new_knob():
    """AC-6: the proportional rule must key off the EXISTING
    silence_min_speech_ratio — a second ratio knob was explicitly rejected."""
    permissive = AudioSettings(silence_min_speech_ratio=1.0)
    assert _required_speech_hops(4, permissive) == 4


# --- AC 4: fail-open ladder, rung by rung ---------------------------------


def test_analyze_vad_returns_none_when_library_unavailable(tmp_path, monkeypatch):
    """AC-4(a): no binary -> abstain. The single most common real-world path
    (every non-Windows platform, every un-fetched checkout)."""
    monkeypatch.setattr(vad_module, "resolve_ten_vad_lib", lambda: None)
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    assert analyze_vad(path, AudioSettings()) is None


def test_analyze_vad_returns_none_when_library_fails_to_load(tmp_path, monkeypatch):
    """AC-4(c): a present-but-unloadable binary (wrong arch, corrupt) must
    abstain and be logged once, never raise."""
    bogus = tmp_path / "not_a_library.dll"
    bogus.write_bytes(b"definitely not a shared library")
    monkeypatch.setattr(vad_module, "resolve_ten_vad_lib", lambda: bogus)
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    assert analyze_vad(path, AudioSettings()) is None


class _BenignLibrary:
    """A library that always succeeds and always scores non-speech.

    Lets the AC-4(b)/(d) fail-open rungs — which are DECODE-path and
    floor-path claims, not model claims — run on checkouts without the
    vendored binary (every CI run today). Same stubbing pattern as the
    ctypes-failure tests below; the real model would add nothing here,
    because these inputs never reach it.
    """

    def create(self, threshold):
        return ctypes.c_void_p(1)

    def process(self, handle, hop):
        return 0.0

    def destroy(self, handle):
        pass


def test_analyze_vad_returns_none_for_corrupt_file(tmp_path, monkeypatch):
    """AC-4(b): a file libsndfile cannot decode -> abstain, never
    is_silent=True (same fixture as the energy guard's own test)."""
    monkeypatch.setattr(vad_module, "_get_library", _BenignLibrary)
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is not a real audio file, just garbage bytes" * 4)

    assert analyze_vad(path, AudioSettings()) is None


def test_analyze_vad_returns_none_for_m4a_container_stub(tmp_path, monkeypatch):
    """AC-4(b), the load-bearing case: /pipeline/process-file accepts
    .m4a/.webm which libsndfile cannot open. Treating "undecodable" as
    "silent" would silently break that tab."""
    monkeypatch.setattr(vad_module, "_get_library", _BenignLibrary)
    path = tmp_path / "clip.m4a"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    assert analyze_vad(path, AudioSettings()) is None


def test_analyze_vad_returns_none_below_min_analysis_floor(tmp_path, monkeypatch):
    """AC-4(d): under silence_min_analysis_ms (100ms) of decoded audio there
    is not enough signal for ANY detector to judge — abstain."""
    monkeypatch.setattr(vad_module, "_get_library", _BenignLibrary)
    path = _write_wav(tmp_path / "tiny.wav", np.random.uniform(-0.1, 0.1, 400))

    assert analyze_vad(path, AudioSettings()) is None


def test_analyze_vad_returns_none_when_process_call_fails(tmp_path, monkeypatch):
    """AC-4(c): a nonzero return code from the C API mid-scan must fail open,
    not propagate. Simulated at the wrapper boundary so it works with or
    without the real binary."""
    class _ExplodingLibrary:
        def create(self, threshold):
            return ctypes.c_void_p(1)

        def process(self, handle, hop):
            raise RuntimeError("ten_vad_process failed (rc=-1)")

        def destroy(self, handle):
            pass

    monkeypatch.setattr(vad_module, "_get_library", lambda: _ExplodingLibrary())
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    assert analyze_vad(path, AudioSettings()) is None


def test_analyze_vad_returns_none_when_create_fails(tmp_path, monkeypatch):
    """AC-4(c): a failed handle creation is equally non-fatal."""
    class _UncreatableLibrary:
        def create(self, threshold):
            raise RuntimeError("ten_vad_create failed (rc=-1)")

    monkeypatch.setattr(vad_module, "_get_library", lambda: _UncreatableLibrary())
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    assert analyze_vad(path, AudioSettings()) is None


def test_failed_library_load_is_cached_not_retried_per_call(tmp_path, monkeypatch):
    """The failed-load sentinel must be cached: a missing binary should log
    ONCE at import-ish time, not once per dictation."""
    calls = {"n": 0}

    def _counting_resolve():
        calls["n"] += 1
        return None

    monkeypatch.setattr(vad_module, "resolve_ten_vad_lib", _counting_resolve)
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    for _ in range(3):
        assert analyze_vad(path, AudioSettings()) is None

    assert calls["n"] == 1


# --- AC 5: streaming + early exit -----------------------------------------


@_requires_dll
def test_analyze_vad_streams_never_reads_whole_file(tmp_path, monkeypatch):
    """AC-5: streaming via soundfile.blocks(), never sf.read() of the whole
    file — memory must not scale with upload length. A regression back to
    sf.read() is caught by making sf.read() itself raise.

    The sf.read() poison alone proves less than it looks: sf.blocks() never
    calls the module-level sf.read() in any implementation, so a regression
    to reading through a SoundFile handle would slip past it. Spying
    sf.blocks() and asserting it is the entry point ACTUALLY used closes the
    near half of that gap. The deeper class weakness is inherited from 029's
    identical pattern; the memory guarantee's real teeth are the early-exit
    consumed<total test below."""
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.3, 0.3, 16000 * 2))

    def _boom(*a, **kw):
        raise AssertionError("analyze_vad must not call sf.read()")

    monkeypatch.setattr(sf, "read", _boom)

    seen = {"blocks": 0}
    real_blocks = sf.blocks

    def _spying_blocks(*args, **kwargs):
        seen["blocks"] += 1
        yield from real_blocks(*args, **kwargs)

    monkeypatch.setattr(sf, "blocks", _spying_blocks)

    result = analyze_vad(path, AudioSettings())

    assert result is not None
    assert isinstance(result, VadAnalysis)
    assert seen["blocks"] == 1, "analyze_vad did not decode via soundfile.blocks()"


@_requires_dll
@_requires_sample
def test_analyze_vad_exits_early_once_verdict_is_certain(tmp_path):
    """AC-5: a 10s file whose speech is entirely in the first second must
    NOT be decoded to the end — once required_speech_hops is met the verdict
    cannot change, so the remaining blocks are wasted work. Counted via a
    wrapping iterator over soundfile.blocks()."""
    speech = _speech_16k_mono()[16000 * 30:16000 * 31]
    padded = np.concatenate([speech, np.zeros(16000 * 9, dtype=np.float32)])
    path = _write_wav(tmp_path / "front_loaded.wav", padded)

    consumed = {"blocks": 0}
    real_blocks = sf.blocks

    def _counting_blocks(*args, **kwargs):
        for block in real_blocks(*args, **kwargs):
            consumed["blocks"] += 1
            yield block

    # analyze_vad imports soundfile lazily inside the function, so patching
    # the attribute on the module object is what the lazy import resolves to.
    sf.blocks = _counting_blocks
    try:
        result = analyze_vad(path, AudioSettings())
    finally:
        sf.blocks = real_blocks

    assert result is not None
    assert result.is_silent is False
    # 1s blocks over a 10s file: a full scan is 10, early exit lands far below.
    assert consumed["blocks"] < 10, (
        f"analyze_vad consumed {consumed['blocks']}/10 blocks — early exit did not fire"
    )


# --- AC 3 / AC 7: contract types and settings -----------------------------


@_requires_dll
def test_vad_analysis_fields_are_native_python_types(tmp_path):
    """AC-3: numpy scalar types are not `is`-identical to Python builtins,
    which silently breaks `is False`-style assertions downstream (the exact
    pin spec 029 added for the energy guard)."""
    path = _write_wav(tmp_path / "zero.wav", np.zeros(16000))

    result = analyze_vad(path, AudioSettings())

    assert result is not None
    assert type(result.is_silent) is bool
    assert type(result.speech_frame_count) is int
    assert type(result.total_frame_count) is int
    assert type(result.max_probability) is float


@_requires_dll
def test_digital_silence_is_silent(tmp_path):
    """The floor case: 3s of true digital silence scores zero speech hops."""
    path = _write_wav(tmp_path / "zero.wav", np.zeros(16000 * 3))

    result = analyze_vad(path, AudioSettings())

    assert result is not None
    assert result.is_silent is True
    assert result.speech_frame_count == 0


@_requires_dll
@_requires_sample
def test_probability_setting_override_flips_verdict(tmp_path):
    """AC-7: silence_vad_probability is a REAL knob — the same input flips
    verdict across it, so the kill-switch/tuning story is not decorative."""
    speech = _speech_16k_mono()[16000 * 30:16000 * 31]
    path = _write_wav(tmp_path / "speech.wav", speech)

    default_result = analyze_vad(path, AudioSettings())
    assert default_result is not None
    assert default_result.is_silent is False
    # An unreachable threshold: no hop can score >= 1.01, so nothing counts
    # as speech and the verdict must flip.
    strict = AudioSettings(silence_vad_probability=1.0, silence_vad_min_speech_frames=5)
    strict_result = analyze_vad(path, strict)
    assert strict_result is not None
    assert strict_result.is_silent is True


def test_vad_settings_defaults_and_env_override(monkeypatch):
    """AC-7: the three new fields exist with the documented defaults and are
    genuinely env-overridable via JUSTSAY_AUDIO_*."""
    defaults = AudioSettings()
    assert defaults.silence_vad_enabled is True
    assert defaults.silence_vad_probability == 0.5
    assert defaults.silence_vad_min_speech_frames == 5

    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_VAD_ENABLED", "false")
    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_VAD_PROBABILITY", "0.8")
    monkeypatch.setenv("JUSTSAY_AUDIO_SILENCE_VAD_MIN_SPEECH_FRAMES", "9")
    overridden = AudioSettings()
    assert overridden.silence_vad_enabled is False
    assert overridden.silence_vad_probability == 0.8
    assert overridden.silence_vad_min_speech_frames == 9


# --- AC 11 / 12 / 13 / 14: behavioural claims against the real model ------


# AC 11(c): the single tolerated net-new discard (a window the 033 pipeline
# eats that shipped-029 kept), pinned by cell AND identity so a *different*
# net-new window, or a second one, fails the suite instead of hiding inside a
# count. Measured at implementation time — see plan.md Deviation 2.
_PINNED_NET_NEW = (200, -12.0, 30)
_PINNED_NET_NEW_START_SECONDS = 240.55


@_requires_dll
@_requires_sample
@pytest.mark.slow
def test_real_speech_is_never_discarded(tmp_path):
    """AC-11, the load-bearing false-positive gate, as a single sweep serving
    all three bars (the sweep is ~960 real-model inferences — running it twice
    bought nothing).

    Spec 029's own rng(7) window-selection machinery, 80 speech-bearing
    windows per cell, at 200/300/500/1000ms and 0/-12/-20dB, judged by the
    SHIPPED decision rule (VAD authoritative, energy fallback):

      (a) 0 dB — native capture level — zero discards, absolute.
      (b) -20 dB — the energy guard's documented failure regime — strictly
          fewer discards than energy in EVERY duration cell.
      (c) whole sweep — total < energy/5, and at most one net-new discard
          vs shipped-029, with that window's identity pinned.

    Bars (a)-(c) are asserted separately on purpose: a failure must name which
    one broke, not just that "the sweep regressed"."""
    audio = _speech_16k_mono()
    settings = AudioSettings()

    shipped_discards: dict[tuple[int, float], list[int]] = {}
    energy_discards: dict[tuple[int, float], list[int]] = {}
    net_new: list[tuple[int, float, int, float]] = []

    for duration_ms in (200, 300, 500, 1000):
        windows = _speech_bearing_windows_with_starts(audio, duration_ms)
        for gain_db in (0.0, -12.0, -20.0):
            cell = (duration_ms, gain_db)
            shipped_discards[cell] = []
            energy_discards[cell] = []
            for i, (start, window) in enumerate(windows):
                attenuated = (window * (10 ** (gain_db / 20))).astype(np.float32)
                path = _write_wav(tmp_path / f"w_{duration_ms}_{gain_db}_{i}.wav", attenuated)
                energy = analyze_silence(path, settings)
                vad = analyze_vad(path, settings)
                energy_silent = energy is not None and energy.is_silent
                # The shipped decision rule: VAD decides whenever it produced
                # a verdict; energy is the fallback when it abstained.
                shipped_silent = vad.is_silent if vad is not None else energy_silent
                if energy_silent:
                    energy_discards[cell].append(i)
                if shipped_silent:
                    shipped_discards[cell].append(i)
                    if not energy_silent:
                        net_new.append((duration_ms, gain_db, i, start / 16000))

    energy_total = sum(len(v) for v in energy_discards.values())
    shipped_total = sum(len(v) for v in shipped_discards.values())

    assert energy_total > 50, (
        "sanity: the energy guard is expected to misfire heavily at -20dB "
        f"(029's documented residual zone) but discarded only {energy_total}"
    )

    # (a) Native capture level: no acceptable rate of eating normal speech.
    native = {
        cell: idx for cell, idx in shipped_discards.items() if cell[1] == 0.0 and idx
    }
    assert native == {}, (
        f"AC 11(a) VIOLATED — {sum(len(v) for v in native.values())} window(s) discarded "
        f"at 0 dB, where the bar is absolute zero: {native}"
    )

    # (b) The quiet regime, per cell: this is the whole reason -20 dB is swept.
    quiet_losses = {
        cell: (len(shipped_discards[cell]), len(energy_discards[cell]))
        for cell in shipped_discards
        if cell[1] == -20.0 and len(shipped_discards[cell]) >= len(energy_discards[cell])
    }
    assert quiet_losses == {}, (
        "AC 11(b) VIOLATED — the VAD pipeline must discard strictly fewer windows than "
        f"the energy guard in every -20 dB cell; (vad, energy) per failing cell: {quiet_losses}"
    )

    # (c) Whole-sweep magnitude, and the net-new set by identity, not by count.
    assert shipped_total < energy_total / 5, (
        f"AC 11(c) VIOLATED — the VAD pipeline discarded {shipped_total}/960 real-speech "
        f"windows against energy's {energy_total}: regression toward energy-level "
        "false positives"
    )
    assert len(net_new) <= 1, (
        f"AC 11(c) VIOLATED — {len(net_new)} windows are newly eaten vs shipped-029 "
        f"(at most 1 is tolerated): {net_new}"
    )
    if net_new:
        duration_ms, gain_db, index, start_seconds = net_new[0]
        assert (duration_ms, gain_db, index) == _PINNED_NET_NEW, (
            "AC 11(c) VIOLATED — the net-new discard is a DIFFERENT window than the one "
            f"pinned at triage {_PINNED_NET_NEW}: got {(duration_ms, gain_db, index)} "
            f"(t={start_seconds:.2f}s). A pinned corner case is acceptable; a shifting "
            "one means the failure class moved."
        )
        assert abs(start_seconds - _PINNED_NET_NEW_START_SECONDS) < 0.01, (
            f"AC 11(c) VIOLATED — pinned window offset drifted: t={start_seconds:.2f}s "
            f"vs {_PINNED_NET_NEW_START_SECONDS}s (sample or rng seed changed?)"
        )


@_requires_dll
@_requires_sample
@pytest.mark.asyncio
async def test_averted_energy_false_positive_is_not_discarded_end_to_end(tmp_path):
    """AC-12: the concrete user-visible improvement over spec 029, pinned
    through the REAL pipeline, not through the two detectors side by side.

    The FIRST window in 029's own rng(7) -20dB/200ms sweep that the energy
    guard wrongly discards is index 2, starting at sample 5_487_905
    (t=342.99s) of the 16kHz-mono decode. Energy calls it silent (peak
    -40.0 dBFS, 0/7 speech frames); the VAD finds 9/12 speech hops at
    max_prob 0.82 and rescues it.

    The two premise asserts stay, then `process_audio` runs with the real
    DLL live and NO `analyze_vad` patch — this file is outside
    `test_pipeline.py`, so its autouse `_vad_abstains` fixture does not
    apply here and a genuine VAD verdict reaches `service.py`'s branch.
    Only the STT provider and the side effects are mocked."""
    audio = _speech_16k_mono()
    start = 5_487_905
    window = audio[start:start + int(16000 * 0.200)]
    attenuated = (window * (10 ** (-20.0 / 20))).astype(np.float32)
    path = _write_wav(tmp_path / "averted_fp.wav", attenuated)

    energy = analyze_silence(path, AudioSettings())
    vad = analyze_vad(path, AudioSettings())

    assert energy is not None and energy.is_silent is True, (
        "premise check: this window is the energy guard's documented false positive"
    )
    assert vad is not None
    assert vad.is_silent is False, (
        f"the VAD must rescue this real-speech window "
        f"(hops={vad.speech_frame_count}/{vad.total_frame_count}, "
        f"max_prob={vad.max_probability:.3f})"
    )

    # Guard the whole point of this test: if some future refactor stubs the
    # VAD out from under it, the pipeline half below would pass vacuously.
    assert vad_module.resolve_ten_vad_lib() is not None
    assert service_module.analyze_vad is analyze_vad, (
        "the pipeline must call the REAL analyze_vad here — AC 12 is about the "
        "genuine VAD verdict flowing through service.py, not a mocked one"
    )

    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="тихе мовлення"))
    stt.model_name = "mock/provider"
    stt.is_local = False

    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)), \
            patch("app.pipeline.service.pyperclip.copy"), \
            patch("app.pipeline.service.save_entry"):
        result = await process_audio(path, language="uk", style="normal")

    assert result.discarded_reason is None, (
        "AC 12 VIOLATED — the pipeline discarded the window the VAD rescued"
    )
    assert result.text == "тихе мовлення"
    stt.transcribe.assert_awaited_once()


def _normalized_to_rms_dbfs(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    return (x * (10 ** (target_dbfs / 20) / rms)).astype(np.float32)


@_requires_dll
@pytest.mark.parametrize("kind", ["white_noise", "square_hum"])
def test_loud_non_speech_is_discarded(tmp_path, kind):
    """AC-13(a)(b): the class the energy guard structurally CANNOT catch.
    Each clip is loud enough to clear the energy gate — proving the hole —
    and must still be discarded by the VAD, proving the hole is closed."""
    sr, seconds = 16000, 3
    n = sr * seconds
    t = np.arange(n) / sr
    if kind == "white_noise":
        signal = _normalized_to_rms_dbfs(np.random.default_rng(7).normal(0, 1, n), -20.0)
    else:
        signal = _normalized_to_rms_dbfs(np.sign(np.sin(2 * np.pi * 60 * t)), -20.0)

    path = _write_wav(tmp_path / f"{kind}.wav", signal)

    energy = analyze_silence(path, AudioSettings())
    vad = analyze_vad(path, AudioSettings())

    assert energy is not None and energy.is_silent is False, (
        f"premise check: {kind} at -20 dBFS RMS must PASS the energy gate — that is the hole"
    )
    assert vad is not None
    assert vad.is_silent is True, (
        f"{kind} reached the model: hops={vad.speech_frame_count}/{vad.total_frame_count}, "
        f"max_prob={vad.max_probability:.3f}"
    )


@_requires_dll
@pytest.mark.xfail(
    strict=True,
    reason=(
        "PINNED DELIBERATE RESIDUAL, decided at triage — not a pending question. "
        "AC 13(c) anticipated a click train scoring above 0.5 on '1-2 hops'; measured "
        "reality on the real model is 22/187 hops clearing 0.5 (max_prob 0.71) against "
        "required_speech_hops=5, because the model smears each 2ms impulse across ~3 "
        "hops. No hop-count rule separates that from quiet real speech, and every "
        "closing mechanism (consecutive-run, higher threshold, level normalisation) "
        "trades against AC 11 — the higher-priority invariant. Deferred as a documented "
        "Cut: see plan.md 'Cuts deferred to a future spec' → 'Discarding impulsive "
        "non-speech (click trains) pre-model'. This strict xfail is the tripwire: if a "
        "model or verdict-rule change ever flips this behaviour, the suite fails and "
        "forces a deliberate review instead of a silent semantic change."
    ),
)
def test_click_train_escape_is_a_pinned_known_limitation(tmp_path):
    """AC-13(c): 8 impulses (<=2ms each, peaks >= -6 dBFS) over 3s of silence
    — a keyboard-click signature that clears the energy gate on peak, and
    which the VAD layer is documented NOT to catch."""
    sr, seconds = 16000, 3
    n = sr * seconds
    signal = np.zeros(n, dtype=np.float32)
    for k in range(8):
        start = int((k + 0.5) * n / 8)
        signal[start:start + 32] = 0.6 * np.hanning(32)

    path = _write_wav(tmp_path / "clicks.wav", signal)

    vad = analyze_vad(path, AudioSettings())

    assert vad is not None
    assert vad.is_silent is True, (
        f"click train reached the model: hops={vad.speech_frame_count}/"
        f"{vad.total_frame_count}, max_prob={vad.max_probability:.3f}"
    )


@_requires_dll
def test_analyze_vad_latency_on_short_clip(tmp_path):
    """AC-14: a 3s/16kHz mono WAV must complete well inside 200ms. A
    generous ceiling against pathological regressions, not a target —
    measured ~18ms at implementation time (see plan.md Deviations)."""
    n = 16000 * 3
    signal = _normalized_to_rms_dbfs(np.random.default_rng(11).normal(0, 1, n), -20.0)
    path = _write_wav(tmp_path / "three_seconds.wav", signal)

    analyze_vad(path, AudioSettings())  # warm the cached library handle
    started = time.perf_counter()
    analyze_vad(path, AudioSettings())
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms <= 200.0, f"analyze_vad took {elapsed_ms:.1f}ms on a 3s clip"


@_requires_dll
@_requires_sample
def test_analyze_vad_latency_on_full_real_sample():
    """AC-14: the full ~6.4min real sample must complete inside 3s. Early
    exit makes this far cheaper than the energy pass over the same file
    (measured ~16ms vs ~1707ms) — speech at the front means the scan stops
    almost immediately."""
    analyze_vad(_TRAIN_AUDIO_MP3, AudioSettings())  # warm
    started = time.perf_counter()
    result = analyze_vad(_TRAIN_AUDIO_MP3, AudioSettings())
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert result is not None
    assert result.is_silent is False
    assert elapsed_ms <= 3000.0, f"analyze_vad took {elapsed_ms:.1f}ms on the full sample"


# --- AC 14(b)(c): contention and concurrency ------------------------------
#
# These exist because AC 14's original bounds were ALL single-threaded, and a
# single-threaded measurement is structurally blind to a lock convoy: the
# whole-scan lock measured 15.2ms solo and 1628.9ms under contention. Solo
# numbers cannot catch that class, so these tests are genuinely concurrent.


def _hum_wav(path: Path, seconds: int) -> Path:
    """A 60Hz square hum at -20 dBFS RMS — loud non-speech that scores ZERO
    speech hops, so it can never early-exit and always pays for a full scan.
    White noise is the wrong choice here: over thousands of hops it
    accumulates occasional >=0.5 hops and exits early (plan Deviation 4)."""
    n = 16000 * seconds
    t = np.arange(n) / 16000
    return _write_wav(path, _normalized_to_rms_dbfs(np.sign(np.sin(2 * np.pi * 60 * t)), -20.0))


@_requires_dll
@pytest.mark.slow
def test_short_clip_is_not_blocked_behind_a_long_scan(tmp_path):
    """AC-14(b): the lock-convoy regression pin.

    Reproduces the real scenario: a long recording is being scanned (Files
    tab / Project Memory upload) when the user hits the dictation hotkey.
    With the lock held across the whole scan this measured 1628.9ms against
    a 1-2.5s total Instant Prompt budget — BEFORE STT was even routed. With
    per-library-entry locking the dictation waits on one 16ms hop.

    The background thread loops so the contention window cannot close early
    and let this pass vacuously."""
    long_path = _hum_wav(tmp_path / "long_hum.wav", 300)
    short_path = _hum_wav(tmp_path / "short_hum.wav", 3)

    analyze_vad(short_path, AudioSettings())  # warm the cached library handle

    in_flight = threading.Event()
    stop = threading.Event()

    def _long_scans():
        while not stop.is_set():
            analyze_vad(long_path, AudioSettings())
            in_flight.set()

    worker = threading.Thread(target=_long_scans, daemon=True)
    worker.start()
    try:
        assert in_flight.wait(timeout=60), "the background long scan never got going"
        started = time.perf_counter()
        result = analyze_vad(short_path, AudioSettings())
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        stop.set()
        worker.join(timeout=60)

    assert result is not None
    assert elapsed_ms <= 250.0, (
        f"AC 14(b) VIOLATED — a 3s clip took {elapsed_ms:.1f}ms while a 300s scan was "
        "running (ceiling 250ms = the 200ms solo bound + 50ms contention allowance). "
        "The library lock is convoying: it must wrap individual library entries only, "
        "never the scan."
    )


@_requires_dll
@_requires_sample
@pytest.mark.slow
def test_concurrent_verdicts_match_single_threaded_verdicts(tmp_path):
    """AC-14(c): coexisting per-call handles must not corrupt each other.

    Four barrier-started concurrent calls over known-verdict inputs must
    return exactly what they return alone. If the per-hop lock release ever
    let two scans interleave inside one handle's streaming state, verdicts
    would drift — this is the behavioural pin on that assumption."""
    speech = _speech_16k_mono()[16000 * 30:16000 * 33]
    inputs = {
        "speech": _write_wav(tmp_path / "c_speech.wav", speech),
        "hum": _hum_wav(tmp_path / "c_hum.wav", 3),
        "noise": _write_wav(
            tmp_path / "c_noise.wav",
            _normalized_to_rms_dbfs(np.random.default_rng(3).normal(0, 1, 16000 * 3), -20.0),
        ),
        "silence": _write_wav(tmp_path / "c_silence.wav", np.zeros(16000 * 3)),
    }

    solo = {name: analyze_vad(path, AudioSettings()) for name, path in inputs.items()}
    assert solo["speech"] is not None and solo["speech"].is_silent is False
    for name in ("hum", "noise", "silence"):
        assert solo[name] is not None and solo[name].is_silent is True, (
            f"premise check: {name} must be silent when run alone"
        )

    barrier = threading.Barrier(len(inputs))
    concurrent: dict[str, VadAnalysis | None] = {}

    def _run(name: str, path: Path) -> None:
        barrier.wait(timeout=60)
        concurrent[name] = analyze_vad(path, AudioSettings())

    threads = [
        threading.Thread(target=_run, args=(name, path)) for name, path in inputs.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    for name in inputs:
        assert concurrent[name] is not None, f"AC 14(c) VIOLATED — {name} abstained concurrently"
        assert concurrent[name].is_silent is solo[name].is_silent, (
            f"AC 14(c) VIOLATED — {name}'s verdict changed under concurrency: "
            f"solo is_silent={solo[name].is_silent}, concurrent={concurrent[name].is_silent}"
        )
        assert concurrent[name].speech_frame_count == solo[name].speech_frame_count, (
            f"AC 14(c) VIOLATED — {name}'s speech-hop count changed under concurrency: "
            f"solo={solo[name].speech_frame_count}, "
            f"concurrent={concurrent[name].speech_frame_count}"
        )


def test_cold_cache_stampede_loads_the_library_exactly_once(tmp_path, monkeypatch):
    """AC-14(c), the `_get_library` race pin — deliberately NOT DLL-gated.

    Four barrier-started calls on a cold cache must produce exactly ONE load
    attempt. Measured before the fix: 4 loads for 4 calls, because
    `_get_library` read and wrote the module global without ever taking the
    lock — which also meant each racing thread emitted its own copy of the
    "logged once, not spammed per dictation" WARNING."""
    loads = {"n": 0}
    load_lock = threading.Lock()

    class _CountingLibrary:
        def __init__(self, lib_path):
            with load_lock:
                loads["n"] += 1
            # Widen the race window: a genuinely unsynchronised _get_library
            # would have all four threads inside this constructor at once.
            time.sleep(0.05)

        def create(self, threshold):
            return ctypes.c_void_p(1)

        def process(self, handle, hop):
            return 0.0

        def destroy(self, handle):
            pass

    monkeypatch.setattr(vad_module, "resolve_ten_vad_lib", lambda: tmp_path / "fake.dll")
    monkeypatch.setattr(vad_module, "_TenVadLibrary", _CountingLibrary)
    path = _write_wav(tmp_path / "audio.wav", np.random.uniform(-0.1, 0.1, 16000))

    barrier = threading.Barrier(4)

    def _run() -> None:
        barrier.wait(timeout=30)
        analyze_vad(path, AudioSettings())

    threads = [threading.Thread(target=_run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert loads["n"] == 1, (
        f"AC 14(c) VIOLATED — {loads['n']} library loads for 4 concurrent cold-cache "
        "calls; the cache must be read and written under _library_lock"
    )


# --- Spec 034 AC 7-8: the lazy-energy-fallback gate latency ----------------
#
# These measure the WHOLE pre-model gate through the real `process_audio`,
# not `analyze_vad` in isolation — the point of spec 034 is orchestration, so
# a detector-only measurement would be blind to it. They live in test_vad.py
# rather than test_pipeline.py on purpose: this file is outside that module's
# autouse `_vad_abstains` fixture, so BOTH detectors are real here.
#
# Shipped ordering paid >= 1707ms for the energy pass alone on the full
# sample (spec 033 Deviations, AC 14). Ceilings below are deliberately
# generous against expected ~16-50ms; the recorded medians in plan.md's
# Deviations are what actually show the win.


# A real full scan of 300s of hum measured ~1.7s (spec 033). Anything under
# this is not a scan at all -- it is analyze_vad's fail-open path returning
# immediately, which makes the contention test both vacuous and a hot spin.
_MIN_CONTENTION_SCAN_S = 0.05


async def _median_gate_ms(path: Path, duration: float, runs: int = 5) -> float:
    """Median wall time of `process_audio` over `path`, gate work only.

    Routing, clipboard and history are patched out, and ``audio_duration`` is
    passed explicitly so `detect_duration` never runs — what is left inside
    the timed region is the silence gate plus mock overhead."""
    stt = MagicMock()
    stt.transcribe = AsyncMock(return_value=TranscriptionResult(text="ok"))
    stt.model_name = "mock/provider"
    stt.is_local = False

    samples: list[float] = []
    with patch("app.pipeline.service.get_routed_provider", return_value=(stt, None)), \
            patch("app.pipeline.service.pyperclip.copy"), \
            patch("app.pipeline.service.save_entry"):
        # Warm the cached library handle and the OS file cache; the autouse
        # _reset_library_cache fixture means every test starts cold.
        await process_audio(path, language="uk", style="normal", audio_duration=duration)
        for _ in range(runs):
            started = time.perf_counter()
            await process_audio(
                path, language="uk", style="normal", audio_duration=duration
            )
            samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


@_requires_dll
@_requires_sample
@pytest.mark.slow
@pytest.mark.asyncio
async def test_gate_latency_on_full_real_sample_solo():
    """AC-7: the headline number. The full ~6.4min sample through the real
    pipeline must clear the gate in <= 500ms median of 5.

    Before spec 034 this path paid the energy guard's full-file decode
    (1707ms) and then threw the verdict away, because the VAD's verdict
    outranks it whenever one exists. Now the energy pass is never invoked and
    the VAD early-exits on the speech at the front."""
    median_ms = await _median_gate_ms(_TRAIN_AUDIO_MP3, duration=384.0)

    assert median_ms <= 500.0, (
        f"AC 7 VIOLATED — the pre-model gate took {median_ms:.1f}ms (median of 5) on the "
        "full sample. Shipped-033 ordering paid >=1707ms here; a number in that range "
        "means the energy pass is running again despite a VAD verdict (ADR 020)."
    )


@_requires_dll
@_requires_sample
@pytest.mark.slow
@pytest.mark.asyncio
async def test_gate_latency_on_full_real_sample_under_contention(tmp_path):
    """AC-8: the same measurement while a full scan hogs the library.

    Spec 033's lesson, applied to the gate rather than the detector: a
    single-threaded measurement is structurally blind to a lock convoy (the
    whole-scan lock measured 15.2ms solo and 1628.9ms under contention). The
    background input is 300s of hum, which scores zero speech hops and so can
    never early-exit — the contention window cannot close and let this pass
    vacuously."""
    long_path = _hum_wav(tmp_path / "long_hum.wav", 300)

    in_flight = threading.Event()
    stop = threading.Event()
    fastest_scan = {"s": math.inf}

    def _long_scans():
        while not stop.is_set():
            started = time.perf_counter()
            analyze_vad(long_path, AudioSettings())
            elapsed = time.perf_counter() - started
            fastest_scan["s"] = min(fastest_scan["s"], elapsed)
            in_flight.set()
            # analyze_vad fails open in MICROSECONDS on a library error, which
            # would turn this loop into a hot spin: it would burn a core for
            # the rest of the run and skew the very measurement it exists to
            # create. `stop.wait` doubles as the sleep and the exit check.
            if elapsed < _MIN_CONTENTION_SCAN_S:
                stop.wait(0.05)

    worker = threading.Thread(target=_long_scans, daemon=True)
    worker.start()
    try:
        assert in_flight.wait(timeout=120), "the background long scan never got going"
        median_ms = await _median_gate_ms(_TRAIN_AUDIO_MP3, duration=384.0)
    finally:
        stop.set()
        worker.join(timeout=120)

    assert not worker.is_alive(), (
        "the background scan thread did not stop within 120s — it would leak "
        "into every test that runs after this one"
    )
    assert fastest_scan["s"] >= _MIN_CONTENTION_SCAN_S, (
        f"the background scan returned in {fastest_scan['s'] * 1000:.1f}ms, i.e. it failed "
        "open instead of scanning 300s of hum — there was no contention to measure and "
        "this test would have passed vacuously"
    )

    assert median_ms <= 750.0, (
        f"AC 8 VIOLATED — the pre-model gate took {median_ms:.1f}ms (median of 5) on the "
        "full sample while a 300s scan was in flight (ceiling 750ms = AC 7's 500ms plus "
        "a 250ms contention allowance)."
    )


@_requires_dll
@_requires_sample
@pytest.mark.slow
@pytest.mark.asyncio
async def test_gate_latency_on_dictation_length_clip(tmp_path):
    """AC-8's second half: the reorder must not have regressed dictation.

    A 3s speech-bearing slice of the real sample — the Instant Prompt shape —
    held to spec 033's own AC 14(b) ceiling of 250ms. Dictation always paid
    the smaller energy cost, so this is the guard against the reorder
    trading a file-upload win for a dictation loss."""
    audio = _speech_16k_mono()
    path = _write_wav(tmp_path / "dictation.wav", audio[:16000 * 3])

    median_ms = await _median_gate_ms(path, duration=3.0)

    assert median_ms <= 250.0, (
        f"AC 8 VIOLATED — a 3s dictation-length clip took {median_ms:.1f}ms (median of 5) "
        "through the pre-model gate, against spec 033's own 250ms bound."
    )
