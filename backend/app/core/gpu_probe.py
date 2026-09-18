"""Vendor-aware GPU probe — priority-ordered, degrade-only detection (ADR 008).

No third-party import (`torch`) and no Windows-only stdlib import (`winreg`)
at module level: both are imported lazily inside the function that needs them,
so this module stays importable on any platform.
"""

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)

_GPU_VENDOR_ENV_VAR = "JUSTSAY_GPU_VENDOR"

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


_cache_lock = threading.Lock()
_cached_result: GpuProbeResult | None = None


def probe_gpu() -> GpuProbeResult:
    """Detect the GPU vendor, name and VRAM; cached for the lifetime of the process.

    Order: `JUSTSAY_GPU_VENDOR` -> `torch.cuda` -> `nvidia-smi` -> Windows registry (AMD/Intel)
    -> `GpuVendor.NONE`. Never raises: a failing source is skipped. `clear_cache()` re-runs it.
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
    `app.embeddings`/`app.stt` for their own provider caches.
    """
    global _cached_result
    with _cache_lock:
        _cached_result = None


def _probe_env_override() -> GpuProbeResult | None:
    """Manual escape hatch for when auto-detect is wrong, and a clean seam
    for tests that need a real end-to-end `probe_gpu()` call without mocking.
    """
    raw = os.environ.get(_GPU_VENDOR_ENV_VAR)
    if not raw:
        return None

    try:
        vendor = GpuVendor(raw.strip().lower())
    except ValueError:
        log.warning("%s=%r is not a valid GPU vendor — ignoring", _GPU_VENDOR_ENV_VAR, raw)
        return None

    return GpuProbeResult(vendor=vendor)


def _probe_torch_cuda() -> GpuProbeResult | None:
    """NVIDIA via torch.cuda — checked before the `nvidia-smi` CLI fallback."""
    try:
        import torch
    except ImportError:
        return None

    try:
        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory

        return GpuProbeResult(
            vendor=GpuVendor.NVIDIA,
            name=props.name,
            vram_total_mb=total // (1024 * 1024),
        )
    except Exception as e:
        log.warning("torch.cuda probe failed: %s", e)
        return None


def _probe_nvidia_smi() -> GpuProbeResult | None:
    """NVIDIA via the `nvidia-smi` CLI — the fallback when torch is absent.

    Runs before any AMD/Intel source, so a secondary adapter cannot misclassify an NVIDIA box. An
    unparseable `memory.total` yields `vram_total_mb=None` and still reports NVIDIA.
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
    name, _, total_str = (part.strip() for part in first_line.partition(","))
    vram_total_mb: int | None
    try:
        vram_total_mb = int(float(total_str))
    except ValueError as e:
        log.warning("nvidia-smi VRAM value malformed: %r (%s)", first_line, e)
        vram_total_mb = None

    return GpuProbeResult(vendor=GpuVendor.NVIDIA, name=name, vram_total_mb=vram_total_mb)


def _probe_windows_registry() -> GpuProbeResult | None:
    """AMD/Intel via the Display Adapters registry class (Windows only).

    Reads `ProviderName`, `DriverDesc` and `HardwareInformation.qwMemorySize` from each numbered
    subkey and returns the classified adapter with the most VRAM — a laptop's discrete card.
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
    """Read one numbered Display Adapters subkey; classify by `ProviderName`.

    Returns `None` for an unclassifiable adapter — a missing value, or a `ProviderName` that is
    neither AMD nor Intel. Every failure degrades to skipping that adapter, never a crash.
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
