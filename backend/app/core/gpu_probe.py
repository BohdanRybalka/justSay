"""Vendor-aware GPU probe — priority-ordered, degrade-only detection.

Replaces the CUDA-only `torch.cuda.is_available()` checks that used to be
duplicated across `app.stt.local`, `app.stt.local_setup`, and
`app.core.router`. Full design rationale, the priority order, and the
Windows-registry-over-WMI-`AdapterRAM` decision (verified live against a real
AMD card) are recorded in `docs/adr/008-gpu-vendor-probe.md`.

No third-party import (`torch`) or Windows-only stdlib import (`winreg`) at
module level — both are imported lazily inside the functions that need them,
mirroring the existing convention in `app.stt.local_factory`.
"""

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)

_ENV_VAR = "JUSTSAY_GPU_VENDOR"

_DISPLAY_ADAPTER_CLASS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


class GpuVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    NONE = "none"


@dataclass(frozen=True)
class GpuProbeResult:
    vendor: GpuVendor
    name: str | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    vram_free_mb: int | None = None


_cache_lock = threading.Lock()
_cached_result: GpuProbeResult | None = None


def probe_gpu() -> GpuProbeResult:
    """Detect the GPU vendor/name/VRAM via a priority-ordered, degrade-only chain.

    Order: `JUSTSAY_GPU_VENDOR` env override -> `torch.cuda` -> `nvidia-smi`
    CLI -> Windows registry (AMD/Intel) -> `GpuVendor.NONE`. Every source
    catches its own failures and returns `None` on any problem; this function
    also wraps each call so a source that raises anyway (rather than
    returning `None`) still can't crash the caller.

    Cached for the lifetime of the process after the first call -- see
    `clear_cache()` to force a fresh probe (production code never needs
    this; it's a test seam for exercising more than one probe outcome, e.g.
    a changed `JUSTSAY_GPU_VENDOR`, within the same process).
    """
    global _cached_result
    with _cache_lock:
        if _cached_result is None:
            _cached_result = _probe_gpu_uncached()
        return _cached_result


def _probe_gpu_uncached() -> GpuProbeResult:
    for source in (
        _probe_env_override,
        _probe_torch_cuda,
        _probe_nvidia_smi,
        _probe_windows_registry,
    ):
        try:
            result = source()
        except Exception as e:
            log.warning("GPU probe source %s failed: %s", source.__name__, e)
            continue
        if result is not None:
            return result

    return GpuProbeResult(vendor=GpuVendor.NONE)


def clear_cache() -> None:
    """Force the next `probe_gpu()` call to re-run the full detection chain.

    Mirrors the `clear_cache()` convention already used by
    `app.llm`/`app.embeddings`/`app.stt` for their own provider caches.
    """
    global _cached_result
    with _cache_lock:
        _cached_result = None


def _probe_env_override() -> GpuProbeResult | None:
    """Manual escape hatch for when auto-detect is wrong, and a clean seam
    for tests that need a real end-to-end `probe_gpu()` call without mocking.
    """
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return None

    try:
        vendor = GpuVendor(raw.strip().lower())
    except ValueError:
        log.warning("%s=%r is not a valid GPU vendor — ignoring", _ENV_VAR, raw)
        return None

    return GpuProbeResult(vendor=vendor)


def _probe_torch_cuda() -> GpuProbeResult | None:
    """NVIDIA via torch.cuda — the only source with a live used/free VRAM split."""
    try:
        import torch
    except ImportError:
        return None

    try:
        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total - reserved

        return GpuProbeResult(
            vendor=GpuVendor.NVIDIA,
            name=props.name,
            vram_total_mb=total // (1024 * 1024),
            vram_used_mb=allocated // (1024 * 1024),
            vram_free_mb=free // (1024 * 1024),
        )
    except Exception as e:
        log.warning("torch.cuda probe failed: %s", e)
        return None


def _probe_nvidia_smi() -> GpuProbeResult | None:
    """NVIDIA via the `nvidia-smi` CLI — total VRAM only, no used/free split.

    Checked before any AMD/Intel source so an NVIDIA box is never
    misclassified by an AMD/Intel-oriented probe finding an unrelated
    secondary adapter.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("nvidia-smi not available: %s", e)
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    first_line = result.stdout.strip().splitlines()[0]
    try:
        name, total_str = (part.strip() for part in first_line.split(",", 1))
        vram_total_mb = int(float(total_str))
    except (ValueError, IndexError) as e:
        log.warning("nvidia-smi output malformed: %r (%s)", first_line, e)
        return None

    return GpuProbeResult(vendor=GpuVendor.NVIDIA, name=name, vram_total_mb=vram_total_mb)


def _probe_windows_registry() -> GpuProbeResult | None:
    """AMD/Intel via the Display Adapters registry class (Windows only).

    Reads each numbered subkey's `ProviderName`/`DriverDesc`/
    `HardwareInformation.qwMemorySize` (a 64-bit value — the 32-bit WMI
    `AdapterRAM` field truncates any card above ~4 GiB, see ADR 008). Takes
    the max-VRAM classified adapter across all subkeys (multi-GPU laptops).
    """
    if os.name != "nt":
        return None

    import winreg

    best: GpuProbeResult | None = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_ADAPTER_CLASS_KEY) as class_key:
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(class_key, index)
                except OSError:
                    break
                index += 1

                candidate = _read_adapter_subkey(class_key, subkey_name)
                if candidate is None:
                    continue
                if best is None or (candidate.vram_total_mb or 0) > (best.vram_total_mb or 0):
                    best = candidate
    except OSError as e:
        log.warning("Windows registry GPU probe failed: %s", e)
        return None

    return best


def _read_adapter_subkey(class_key, subkey_name: str) -> GpuProbeResult | None:
    """Read one numbered Display Adapters subkey; classify by ProviderName.

    Returns `None` for unclassifiable adapters (missing values, non-AMD/Intel
    `ProviderName`, e.g. a software/basic render adapter) — every failure
    degrades to "skip this adapter," never a crash.
    """
    import winreg

    try:
        with winreg.OpenKey(class_key, subkey_name) as adapter_key:
            provider_name, _ = winreg.QueryValueEx(adapter_key, "ProviderName")
            driver_desc, _ = winreg.QueryValueEx(adapter_key, "DriverDesc")
            vram_bytes, _ = winreg.QueryValueEx(adapter_key, "HardwareInformation.qwMemorySize")
    except OSError:
        return None

    vendor = _classify_provider_name(provider_name)
    if vendor is None:
        return None

    try:
        vram_total_mb = int(vram_bytes) // (1024 * 1024)
    except (TypeError, ValueError):
        vram_total_mb = None

    return GpuProbeResult(vendor=vendor, name=driver_desc, vram_total_mb=vram_total_mb)


def _classify_provider_name(provider_name) -> GpuVendor | None:
    """Substring-match `ProviderName` into AMD/Intel; anything else is skipped
    rather than guessed at (driver-version-dependent, see ADR 008's risks)."""
    if not isinstance(provider_name, str):
        return None
    lowered = provider_name.lower()
    if "advanced micro devices" in lowered or "ati technologies" in lowered:
        return GpuVendor.AMD
    if "intel" in lowered:
        return GpuVendor.INTEL
    return None
