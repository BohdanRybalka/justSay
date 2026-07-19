"""Neural voice-activity detection via TEN VAD, loaded through ctypes.

The primary pre-model no-speech detector (spec 033 /
docs/adr/019-ten-vad-neural-silence-gate.md), layered in FRONT of
`app.audio.analysis`'s energy guard rather than replacing it. Energy
thresholding has a blind spot no tuning can close — loud non-speech
(keyboard clicks, breathing, hum, noise) clears a loudness gate by
definition — and a residual false-positive zone on quiet speech. A neural
verdict fixes both directions at once.

Deliberately stdlib + numpy only: the backend ships as a frozen PyInstaller
sidecar whose venv contains only numpy/soundfile/sounddevice, so Silero
(needs onnxruntime/torch) and webrtcvad (pip dep, C extension, pre-neural
quality) are both ruled out. TEN VAD is a prebuilt C library consumed via
`ctypes`, which costs zero new pip dependencies.

EVERY failure path fails OPEN — `analyze_vad` returns ``None`` and never
raises, never reports ``is_silent=True``. A detector that eats the user's
real words is a far worse bug than one that lets a hallucination through;
spec 029 paid two review iterations to learn that, and this module does not
relearn it. ``None`` means "this layer abstains, fall back to the energy
verdict", never "silent".
"""

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Imported as a MODULE, not `from ... import required_speech_units`: the
# delegation below resolves through the module attribute, which is what lets
# a test monkeypatch `analysis.required_speech_units` and prove both
# detectors route through one implementation (spec 034, AC 11).
from app.audio import analysis
from app.audio.config import AudioSettings

log = logging.getLogger(__name__)

# TEN VAD's own default hop: 256 samples @ 16 kHz = 16 ms. A module
# constant, NOT an AudioSettings field -- the library's model is trained for
# this hop, so it is a property of the engine rather than a tuning knob
# (same rationale as analysis.py's _MIN_SPEECH_UNITS_FLOOR).
_HOP_SAMPLES = 256
_VAD_SAMPLE_RATE = 16000

# 1 s decode blocks -- streaming, so memory stays flat regardless of upload
# length (a 6-minute meeting upload must not be read into RAM whole).
_BLOCK_SECONDS = 1.0

_ENV_OVERRIDE = "JUSTSAY_TEN_VAD_LIB"


@dataclass(frozen=True)
class VadAnalysis:
    speech_frame_count: int
    total_frame_count: int
    max_probability: float
    is_silent: bool


def _platform_lib_name() -> str:
    if sys.platform == "win32":
        return "ten_vad.dll"
    if sys.platform == "darwin":
        return "libten_vad.dylib"
    return "libten_vad.so"


def resolve_ten_vad_lib() -> Path | None:
    """Locate the TEN VAD shared library, or ``None`` when unavailable.

    Degrade-only chain mirroring `local_vulkan_cmd.resolve_binary_path()`:
    env override -> frozen bundle -> dev vendor dir -> ``None``. Each source
    is accepted only when the resolved file actually EXISTS, so a stale env
    var pointing at a deleted file falls through to the next source instead
    of hard-failing the dictation.

    ``None`` is a normal, expected outcome — every non-Windows platform and
    every checkout that hasn't run `backend/scripts/fetch_ten_vad.py`. The
    caller degrades to the energy guard alone.
    """
    lib_name = _platform_lib_name()

    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        log.warning(
            "%s points at %s which does not exist — falling through to bundled/vendored lookup",
            _ENV_OVERRIDE, candidate,
        )

    # PyInstaller onedir: sys._MEIPASS is the _internal/ directory the
    # release workflow ships (build_sidecar.spec places the DLL under
    # ten_vad/ there).
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidate = Path(meipass) / "ten_vad" / lib_name
            if candidate.is_file():
                return candidate

    candidate = Path(__file__).resolve().parents[2] / "vendor" / "ten-vad" / lib_name
    if candidate.is_file():
        return candidate

    return None


class _TenVadLibrary:
    """Minimal typed ctypes binding for TEN VAD's C API.

    Signatures confirmed against include/ten_vad.h at the pinned tag (v1.0):
        int ten_vad_create(ten_vad_handle_t *handle, size_t hop_size, float threshold);
        int ten_vad_process(ten_vad_handle_t handle, const int16_t *audio_data,
                            size_t audio_data_length, float *out_probability, int *out_flag);
        int ten_vad_destroy(ten_vad_handle_t *handle);
    All three return 0 on success, -1 on error. ``hop_size`` is in SAMPLES.

    We write our own binding rather than vendoring upstream's example .py --
    that file is not part of the pinned artifact set and would be a second
    thing to keep in sync.
    """

    def __init__(self, lib_path: Path) -> None:
        self._lib = ctypes.CDLL(str(lib_path))

        self._lib.ten_vad_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_float,
        ]
        self._lib.ten_vad_create.restype = ctypes.c_int

        self._lib.ten_vad_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.ten_vad_process.restype = ctypes.c_int

        self._lib.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.ten_vad_destroy.restype = ctypes.c_int

    def create(self, threshold: float) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        rc = self._lib.ten_vad_create(
            ctypes.byref(handle), ctypes.c_size_t(_HOP_SAMPLES), ctypes.c_float(threshold)
        )
        if rc != 0 or not handle.value:
            raise RuntimeError(f"ten_vad_create failed (rc={rc})")
        return handle

    def process(self, handle: ctypes.c_void_p, hop: np.ndarray) -> float:
        probability = ctypes.c_float()
        flag = ctypes.c_int()
        rc = self._lib.ten_vad_process(
            handle,
            hop.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(_HOP_SAMPLES),
            ctypes.byref(probability),
            ctypes.byref(flag),
        )
        if rc != 0:
            raise RuntimeError(f"ten_vad_process failed (rc={rc})")
        return float(probability.value)

    def destroy(self, handle: ctypes.c_void_p) -> None:
        self._lib.ten_vad_destroy(ctypes.byref(handle))


# The CDLL handle is loaded once per process and cached. A FAILED load is
# cached too (as the _LOAD_FAILED sentinel) so a missing/broken binary logs
# once at WARNING instead of once per dictation. EVERY read and write of the
# cache happens under this lock -- without it, a cold-cache stampede loads
# the DLL once per racing thread and emits one WARNING each.
#
# The same lock serialises entry into the library, but only ONE ENTRY AT A
# TIME: ten_vad_create, each per-hop ten_vad_process, ten_vad_destroy.
# Decode/resample/carry work stays outside it. Upstream documents nothing
# about thread-safety (README and include/ten_vad.h at the pinned tag are
# both silent), so strict mutual exclusion of library entries is preserved --
# but a concurrent caller now waits at most one 16 ms-hop inference instead
# of a whole file scan (measured: 1628.9 ms behind a 300 s scan when the lock
# spanned the scan).
class _LoadFailed:
    """Sentinel type for a cached failed load.

    A plain ``object()`` would force the cache annotation to include ``object``,
    which subsumes every other union member and makes the type meaningless to
    a checker. A dedicated class keeps the union assertive.
    """


_LOAD_FAILED = _LoadFailed()
_library: _TenVadLibrary | _LoadFailed | None = None
_library_lock = threading.Lock()


def _get_library() -> _TenVadLibrary | None:
    global _library
    with _library_lock:
        if _library is _LOAD_FAILED:
            return None
        if isinstance(_library, _TenVadLibrary):
            return _library

        lib_path = resolve_ten_vad_lib()
        if lib_path is None:
            log.info(
                "TEN VAD library not found (no env override, no bundled or vendored copy) — "
                "neural VAD disabled, energy guard alone decides. "
                "Run backend/scripts/fetch_ten_vad.py to enable it."
            )
            _library = _LOAD_FAILED
            return None

        try:
            _library = _TenVadLibrary(lib_path)
            log.info("TEN VAD library loaded from %s", lib_path)
            return _library
        except Exception as e:
            log.warning(
                "TEN VAD library at %s could not be loaded — failing open, energy guard alone "
                "decides: %s", lib_path, e,
            )
            _library = _LOAD_FAILED
            return None


def _reset_library_cache() -> None:
    """Test-only: drop the cached load so a monkeypatched resolver takes effect."""
    global _library
    with _library_lock:
        _library = None


def _required_speech_hops(total_hop_count: int, settings: AudioSettings) -> int:
    """Length-proportional speech-hop requirement.

    The VAD's 16 ms hops, capped by ``silence_vad_min_speech_frames`` — the
    one field distinguishing this from the energy guard's frame requirement.
    Reuses the SHIPPED ``silence_min_speech_ratio`` (0.15) rather than
    minting a new knob; the rule itself lives in
    `analysis.required_speech_units` — see its docstring for the rationale.
    """
    return analysis.required_speech_units(
        total_hop_count,
        cap=settings.silence_vad_min_speech_frames,
        ratio=settings.silence_min_speech_ratio,
    )


def _to_mono_16k(block: np.ndarray, samplerate: int) -> np.ndarray:
    """Mean-collapse to mono and linear-resample to 16 kHz.

    The recorder path is already 16 kHz mono (``AudioSettings.sample_rate``),
    so dictation — the latency-sensitive case — skips resampling entirely.
    Uploads at 44.1/48 kHz get a linear-interpolation approximation, which is
    adequate for speech-PRESENCE detection; the file handed to the STT
    provider is never touched by any of this.

    Each block is resampled INDEPENDENTLY, which duplicates the boundary
    sample and drifts the time base by ~1 sample per block on non-16 kHz
    input. Harmless for a presence verdict (and never hit on the dictation
    path, which is already 16 kHz), but it means the caller's carry buffer
    stitches approximately-continuous audio, not sample-exact audio.
    """
    mono = block.mean(axis=1) if block.ndim > 1 else block
    if samplerate == _VAD_SAMPLE_RATE or mono.size == 0:
        return mono.astype(np.float32)

    n_target = int(round(mono.size * _VAD_SAMPLE_RATE / samplerate))
    if n_target <= 0:
        return np.empty(0, dtype=np.float32)
    x_orig = np.arange(mono.size, dtype=np.float64)
    x_target = np.linspace(0.0, mono.size - 1, num=n_target, dtype=np.float64)
    return np.interp(x_target, x_orig, mono).astype(np.float32)


def analyze_vad(audio_path: Path, settings: AudioSettings) -> VadAnalysis | None:
    """Stream ``audio_path`` through TEN VAD and decide whether it is silent.

    A 16 ms hop counts as speech when its probability clears
    ``settings.silence_vad_probability`` (0.5 — upstream TEN VAD's and
    Silero's shared reference default, NOT a number fitted to this project's
    single recording). The number of speech hops required scales with clip
    length via `_required_speech_hops`. ``is_silent`` is then simply
    "fewer speech hops than required".

    Returns ``None`` — never raises, never reports ``is_silent=True`` — when:
      - the library is unavailable or fails to load/call (energy-only),
      - the file cannot be decoded (``.m4a``/``.webm`` uploads libsndfile
        can't open — same fail-open rule as `analyze_silence`),
      - fewer than ``settings.silence_min_analysis_ms`` of audio decoded
        (a truncated-but-header-valid WAV yields a handful of samples, which
        is not enough for ANY detector to judge).
    Callers MUST treat ``None`` as "this layer abstains", falling back to the
    energy verdict.

    Streams via ``soundfile.blocks()`` and EXITS EARLY the moment the
    required speech-hop count is met — so the common (speech-bearing) case
    never decodes the whole file, and only genuine silence pays for a full
    scan. That scan replaces a wasted model inference, so it is cheap in the
    only accounting that matters.
    """
    library = _get_library()
    if library is None:
        return None

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        samplerate = info.samplerate
        if not samplerate:
            return None

        # Hop count the file COULD yield if fully decoded -- needed for the
        # length-proportional requirement before early exit can trigger.
        # Derived from the declared duration; the decoded-ms floor below is
        # what actually catches a truncated file.
        estimated_total_hops = max(
            1, int(info.frames * _VAD_SAMPLE_RATE / samplerate) // _HOP_SAMPLES
        )
        required_hops = _required_speech_hops(estimated_total_hops, settings)

        blocksize = max(_HOP_SAMPLES, int(_BLOCK_SECONDS * samplerate))
        carry = np.empty(0, dtype=np.float32)
        speech_hops = 0
        total_hops = 0
        max_probability = 0.0
        total_samples_decoded = 0
        early_exit = False

        # The lock wraps each INDIVIDUAL library entry, never the scan: a
        # concurrent dictation waits on one hop's inference, not on this
        # file's whole decode (AC 14(b)).
        with _library_lock:
            handle = library.create(float(settings.silence_vad_probability))
        try:
            for block in sf.blocks(
                str(audio_path), blocksize=blocksize, dtype="float32", always_2d=True
            ):
                total_samples_decoded += block.shape[0]
                mono = _to_mono_16k(block, samplerate)
                carry = np.concatenate((carry, mono)) if carry.size else mono

                n_hops = carry.size // _HOP_SAMPLES
                for i in range(n_hops):
                    chunk = carry[i * _HOP_SAMPLES:(i + 1) * _HOP_SAMPLES]
                    # int16 conversion matching the C API's expected
                    # sample format; clip first so an over-unity float
                    # sample wraps to a loud click instead of silently
                    # overflowing into garbage the model would score.
                    hop = np.ascontiguousarray(
                        np.clip(chunk, -1.0, 1.0) * 32767.0, dtype=np.int16
                    )
                    with _library_lock:
                        probability = library.process(handle, hop)
                    total_hops += 1
                    max_probability = max(max_probability, probability)
                    if probability >= settings.silence_vad_probability:
                        speech_hops += 1

                carry = carry[n_hops * _HOP_SAMPLES:]

                if speech_hops >= required_hops:
                    early_exit = True
                    break
        finally:
            with _library_lock:
                library.destroy(handle)
    except Exception as e:
        log.warning(
            "Neural VAD could not analyze %s — failing open, energy guard alone decides: %s",
            audio_path, e,
        )
        return None

    # Only meaningful when the whole file was actually consumed: an early
    # exit stops mid-file by design, and that case is already a not-silent
    # verdict, so the floor has nothing to protect against there.
    if not early_exit:
        decoded_ms = (total_samples_decoded / samplerate) * 1000.0
        if decoded_ms < settings.silence_min_analysis_ms:
            log.warning(
                "Neural VAD: only %.1fms of audio decoded from %s (floor=%.1fms) "
                "— failing open, energy guard alone decides",
                decoded_ms, audio_path, settings.silence_min_analysis_ms,
            )
            return None
        # The file is fully decoded, so the estimate is now a fact -- recompute
        # against the real hop count rather than the header-derived guess.
        required_hops = _required_speech_hops(max(1, total_hops), settings)

    # int()/float()/bool(): numpy scalar types are not `is`-identical to
    # Python builtins, which would silently break `is False`-style assertions
    # downstream (same convention as analysis.py).
    return VadAnalysis(
        speech_frame_count=int(speech_hops),
        total_frame_count=int(total_hops),
        max_probability=float(max_probability),
        is_silent=bool(speech_hops < required_hops),
    )
