"""Neural voice-activity detection via TEN VAD, loaded through ctypes.

The primary pre-model no-speech detector, in FRONT of `app.audio.analysis`'s
energy guard rather than replacing it (ADR 019). stdlib and numpy only: TEN
VAD is a prebuilt C library reached through `ctypes`, costing no new pip dep.

EVERY failure path fails OPEN: `analyze_vad` returns ``None`` rather than
raising or reporting silence, and ``None`` means "abstain", not "silent".
"""

import ctypes
import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio import analysis
from app.audio.config import AudioSettings
from app.core.frozen_build import is_frozen_build

log = logging.getLogger(__name__)

_HOP_SAMPLES = 256
_VAD_SAMPLE_RATE = 16000

_BLOCK_SECONDS = 1.0

_ENV_OVERRIDE = "JUSTSAY_TEN_VAD_LIB"


@dataclass(frozen=True)
class VadAnalysis:
    speech_hop_count: int
    total_hop_count: int
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

    Env override, then frozen bundle, then dev vendor dir; each is accepted
    only when the resolved file exists, so a stale override falls through
    rather than failing. ``None`` is normal — the caller degrades to energy.
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

    if is_frozen_build():
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

    Signatures are pinned against ``include/ten_vad.h`` at the vendored tag;
    every entry point returns 0 on success and -1 on error, ``hop_size`` is
    in samples, and `create` and `process` raise on a non-zero rc.
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


class _LoadFailed:
    """Sentinel type for a cached failed load.

    A dedicated class rather than ``object()`` keeps the cache's union type
    assertive; ``object`` would subsume every other member of it.
    """


_LOAD_FAILED = _LoadFailed()
_library: _TenVadLibrary | _LoadFailed | None = None
_library_cache_lock = threading.Lock()
_ten_vad_api_lock = threading.Lock()


def _get_library() -> _TenVadLibrary | None:
    global _library
    with _library_cache_lock:
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
    with _library_cache_lock:
        _library = None


def _required_speech_hops(total_hop_count: int, settings: AudioSettings) -> int:
    """Length-proportional speech-hop requirement for the neural VAD.

    16 ms hops, capped by ``silence_vad_min_speech_frames`` — the one field
    distinguishing this from the energy guard's frame requirement.
    """
    return analysis.required_speech_units(
        total_hop_count,
        cap=settings.silence_vad_min_speech_frames,
        ratio=settings.silence_min_speech_ratio,
    )


def _to_mono_16k(block: np.ndarray, samplerate: int) -> np.ndarray:
    """Mean-collapse to mono and linear-resample to 16 kHz.

    Blocks are resampled independently, so on non-16 kHz input the time base
    drifts about a sample per block: the caller's carry buffer stitches
    approximately-continuous audio, which a presence verdict tolerates.
    """
    mono = analysis.to_mono(block)
    if samplerate == _VAD_SAMPLE_RATE or mono.size == 0:
        return mono

    n_target = int(round(mono.size * _VAD_SAMPLE_RATE / samplerate))
    if n_target <= 0:
        return np.empty(0, dtype=np.float32)
    x_orig = np.arange(mono.size, dtype=np.float64)
    x_target = np.linspace(0.0, mono.size - 1, num=n_target, dtype=np.float64)
    return np.interp(x_target, x_orig, mono).astype(np.float32)


def analyze_vad(audio_path: Path, settings: AudioSettings) -> VadAnalysis | None:
    """Stream ``audio_path`` through TEN VAD and decide whether it is silent.

    Silent means fewer hops clearing ``silence_vad_probability`` than
    `_required_speech_hops` asks for. ``None`` — never a raise, never a silent
    verdict — means this layer abstains and the energy verdict decides.
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

        with _ten_vad_api_lock:
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
                    hop = np.ascontiguousarray(
                        np.clip(chunk, -1.0, 1.0) * 32767.0, dtype=np.int16
                    )
                    with _ten_vad_api_lock:
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
            with _ten_vad_api_lock:
                library.destroy(handle)
    except Exception as e:
        log.warning(
            "Neural VAD could not analyze %s — failing open, energy guard alone decides: %s",
            audio_path, e,
        )
        return None

    if not early_exit:
        decoded_ms = (total_samples_decoded / samplerate) * 1000.0
        if decoded_ms < settings.silence_min_analysis_ms:
            log.warning(
                "Neural VAD: only %.1fms of audio decoded from %s (floor=%.1fms) "
                "— failing open, energy guard alone decides",
                decoded_ms, audio_path, settings.silence_min_analysis_ms,
            )
            return None
        required_hops = _required_speech_hops(max(1, total_hops), settings)

    return VadAnalysis(
        speech_hop_count=int(speech_hops),
        total_hop_count=int(total_hops),
        max_probability=float(max_probability),
        is_silent=bool(speech_hops < required_hops),
    )


def selftest() -> tuple[bool, str]:
    """``--selftest-ten-vad`` backend. Never raises.

    Resolves the library, loads it through ctypes and runs a synthetic
    one-second 16 kHz probe through `analyze_vad`, reporting which of the
    three gave way — the abstention a dead neural gate hides behind.
    """
    library_path = resolve_ten_vad_lib()
    if library_path is None:
        return False, (
            f"{_platform_lib_name()} not found via {_ENV_OVERRIDE}, the frozen "
            "bundle, or backend/vendor/ten-vad — the neural gate is inert"
        )

    try:
        if _get_library() is None:
            return False, f"the library at {library_path} did not load"

        import soundfile as sf

        tone = 0.3 * np.sin(
            2.0 * np.pi * 220.0 * np.arange(_VAD_SAMPLE_RATE, dtype=np.float32)
            / _VAD_SAMPLE_RATE
        )
        with tempfile.TemporaryDirectory() as probe_dir:
            probe_path = Path(probe_dir) / "ten_vad_selftest.wav"
            sf.write(str(probe_path), tone.astype(np.float32), _VAD_SAMPLE_RATE)
            result = analyze_vad(probe_path, AudioSettings())
    except Exception as e:
        return False, f"the selftest raised against {library_path}: {e}"

    if result is None:
        return False, (
            f"the library at {library_path} loaded but abstained on a synthetic "
            "one-second probe clip"
        )
    return True, "ok"
