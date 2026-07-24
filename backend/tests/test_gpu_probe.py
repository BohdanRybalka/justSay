"""Tests for `app.core.gpu_probe`: priority order, degrade-on-failure, and
the no-third-party-import-at-module-level constraint.

Real hardware numbers (AMD Radeon RX 5700 XT, 8GB card) are reproduced from
`docs/adr/008-gpu-vendor-probe.md`'s live verification.
"""

import sys
from unittest.mock import MagicMock

import pytest

from app.core import gpu_probe
from app.core.gpu_probe import GpuProbeResult, GpuVendor

_requires_windows = pytest.mark.skipif(
    sys.platform != "win32",
    reason="winreg is a Windows-only stdlib module; this probe runs only on Windows",
)


@pytest.fixture(autouse=True)
def _clear_env_override(monkeypatch):
    monkeypatch.delenv("JUSTSAY_GPU_VENDOR", raising=False)




def test_module_imports_no_third_party_at_module_level():
    """Importing gpu_probe must not pull in torch or winreg.

    Same style as `test_stt_local_factory.py`'s equivalent module-level
    import-hygiene test.
    """
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "app.core.gpu_probe":
            del sys.modules[name]
    importlib.import_module("app.core.gpu_probe")
    mod = sys.modules["app.core.gpu_probe"]
    assert not hasattr(mod, "torch")
    assert not hasattr(mod, "winreg")




def test_env_override_wins_over_all_auto_detected_sources(monkeypatch):
    monkeypatch.setenv("JUSTSAY_GPU_VENDOR", "amd")
    monkeypatch.setattr(
        gpu_probe, "_probe_torch_cuda",
        lambda: GpuProbeResult(vendor=GpuVendor.NVIDIA, name="fake nvidia"),
    )
    monkeypatch.setattr(
        gpu_probe, "_probe_nvidia_smi",
        lambda: GpuProbeResult(vendor=GpuVendor.NVIDIA, name="fake nvidia-smi"),
    )
    monkeypatch.setattr(
        gpu_probe, "_probe_windows_registry",
        lambda: GpuProbeResult(vendor=GpuVendor.INTEL, name="fake intel"),
    )

    result = gpu_probe.probe_gpu()

    assert result.vendor == GpuVendor.AMD
    assert result.name is None


def test_env_override_case_insensitive_value(monkeypatch):
    monkeypatch.setenv("JUSTSAY_GPU_VENDOR", "NVIDIA")
    result = gpu_probe._probe_env_override()
    assert result == GpuProbeResult(vendor=GpuVendor.NVIDIA)


def test_env_override_invalid_value_is_ignored_and_logged(monkeypatch, caplog):
    monkeypatch.setenv("JUSTSAY_GPU_VENDOR", "not-a-real-vendor")

    import logging

    with caplog.at_level(logging.WARNING, logger="app.core.gpu_probe"):
        result = gpu_probe._probe_env_override()

    assert result is None
    assert "not-a-real-vendor" in caplog.text


def test_nvidia_source_wins_over_windows_registry_amd(monkeypatch):
    """A mocked torch.cuda NVIDIA result must win over a mocked registry AMD
    result, and the registry source must never even be consulted."""
    monkeypatch.setattr(
        gpu_probe, "_probe_torch_cuda",
        lambda: GpuProbeResult(
            vendor=GpuVendor.NVIDIA, name="RTX 4090",
            vram_total_mb=24576, vram_used_mb=1024, vram_free_mb=23552,
        ),
    )
    registry_spy = MagicMock(
        return_value=GpuProbeResult(vendor=GpuVendor.AMD, name="Radeon RX 5700 XT")
    )
    monkeypatch.setattr(gpu_probe, "_probe_windows_registry", registry_spy)

    result = gpu_probe.probe_gpu()

    assert result.vendor == GpuVendor.NVIDIA
    assert result.name == "RTX 4090"
    registry_spy.assert_not_called()


def test_registry_probe_not_invoked_when_not_windows(monkeypatch):
    """`_probe_windows_registry` must bail out before ever importing `winreg`
    when `os.name != 'nt'` — asserted by spying on the import itself, not
    just the return value."""
    monkeypatch.setattr(gpu_probe.os, "name", "posix")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "winreg":
            raise AssertionError("winreg must not be imported when os.name != 'nt'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert gpu_probe._probe_windows_registry() is None




def test_probe_gpu_never_raises_and_degrades_to_none_when_every_source_fails(monkeypatch):
    def _raise_import_error():
        raise ImportError("no torch")

    def _raise_file_not_found():
        raise FileNotFoundError("no nvidia-smi binary")

    def _raise_os_error():
        raise OSError("registry unreadable")

    monkeypatch.setattr(gpu_probe, "_probe_torch_cuda", _raise_import_error)
    monkeypatch.setattr(gpu_probe, "_probe_nvidia_smi", _raise_file_not_found)
    monkeypatch.setattr(gpu_probe, "_probe_windows_registry", _raise_os_error)

    result = gpu_probe.probe_gpu()

    assert result == GpuProbeResult(vendor=GpuVendor.NONE)


def test_torch_cuda_probe_returns_none_when_torch_missing():
    from unittest.mock import patch

    with patch.dict("sys.modules", {"torch": None}):
        assert gpu_probe._probe_torch_cuda() is None


def test_torch_cuda_probe_returns_none_when_cuda_unavailable():
    from unittest.mock import patch

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert gpu_probe._probe_torch_cuda() is None


def test_torch_cuda_probe_computes_vram_from_real_total_memory_attribute():
    """Regression (spec 014, iteration 2): exercises `_probe_torch_cuda()`'s
    real body — not a wholesale mock of `probe_gpu()`/`_probe_torch_cuda`
    itself — with a fake device-properties object restricted (via `spec`) to
    only the attributes the real PyTorch `_CudaDeviceProperties` object has
    (`name`, `total_memory`). Accessing the nonexistent `total_mem` would
    raise `AttributeError` against this fixture, exactly as it does against
    real PyTorch, catching the exact bug that previously shipped: reading
    `total_mem` silently degraded this source to `None` on real NVIDIA
    hardware via this function's own broad `except Exception`.
    """
    from unittest.mock import patch

    fake_props = MagicMock(spec=["name", "total_memory"])
    fake_props.name = "NVIDIA GeForce RTX 3060"
    fake_props.total_memory = 12 * 1024 * 1024 * 1024

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_properties.return_value = fake_props
    fake_torch.cuda.memory_reserved.return_value = 2 * 1024 * 1024 * 1024
    fake_torch.cuda.memory_allocated.return_value = 1 * 1024 * 1024 * 1024

    with patch.dict("sys.modules", {"torch": fake_torch}):
        result = gpu_probe._probe_torch_cuda()

    assert result is not None
    assert result.vendor == GpuVendor.NVIDIA
    assert result.name == "NVIDIA GeForce RTX 3060"
    assert result.vram_total_mb == 12 * 1024
    assert result.vram_used_mb == 1 * 1024
    assert result.vram_free_mb == (12 - 2) * 1024


def test_nvidia_smi_probe_returns_none_when_binary_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(gpu_probe.subprocess, "run", _raise)
    assert gpu_probe._probe_nvidia_smi() is None


def test_nvidia_smi_probe_returns_none_on_malformed_output(monkeypatch):
    fake_result = MagicMock(returncode=0, stdout="some gpu name, not-a-number\n", stderr="")
    monkeypatch.setattr(gpu_probe.subprocess, "run", lambda *a, **k: fake_result)
    assert gpu_probe._probe_nvidia_smi() is None


def test_nvidia_smi_probe_parses_name_and_total_vram(monkeypatch):
    fake_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3060, 12288\n", stderr="")
    monkeypatch.setattr(gpu_probe.subprocess, "run", lambda *a, **k: fake_result)

    result = gpu_probe._probe_nvidia_smi()

    assert result.vendor == GpuVendor.NVIDIA
    assert result.name == "NVIDIA GeForce RTX 3060"
    assert result.vram_total_mb == 12288


@_requires_windows
def test_windows_registry_probe_returns_none_on_openkey_failure(monkeypatch):
    monkeypatch.setattr(gpu_probe.os, "name", "nt")
    import winreg

    def _raise(*args, **kwargs):
        raise OSError("registry class key missing")

    monkeypatch.setattr(winreg, "OpenKey", _raise)
    assert gpu_probe._probe_windows_registry() is None


class _FakeAdapterKey:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@_requires_windows
def test_windows_registry_probe_classifies_amd_and_picks_max_vram_adapter(monkeypatch):
    """Reproduces the ADR-008 live measurement: AMD Radeon RX 5700 XT reports
    HardwareInformation.qwMemorySize = 8573157376 bytes (7.98 GiB, correct) —
    a software render adapter (no real VRAM) must be skipped, and a smaller
    Intel adapter must lose to the larger AMD one."""
    monkeypatch.setattr(gpu_probe.os, "name", "nt")
    import winreg

    adapter_data = {
        "0000": {
            "ProviderName": "Advanced Micro Devices, Inc.",
            "DriverDesc": "AMD Radeon RX 5700 XT",
            "HardwareInformation.qwMemorySize": 8573157376,
        },
        "0001": {
            "ProviderName": "Microsoft Basic Render Driver",
            "DriverDesc": "Microsoft Basic Render Driver",
            "HardwareInformation.qwMemorySize": 0,
        },
        "0002": {
            "ProviderName": "Intel Corporation",
            "DriverDesc": "Intel(R) UHD Graphics",
            "HardwareInformation.qwMemorySize": 1073741824,
        },
    }
    subkey_order = list(adapter_data)

    def fake_open_key(key, subpath):
        if subpath in adapter_data:
            return _FakeAdapterKey(subpath)
        return _FakeAdapterKey("class")

    def fake_enum_key(key, index):
        if index < len(subkey_order):
            return subkey_order[index]
        raise OSError("no more subkeys")

    def fake_query_value_ex(key, value_name):
        return adapter_data[key.name][value_name], 0

    monkeypatch.setattr(winreg, "OpenKey", fake_open_key)
    monkeypatch.setattr(winreg, "EnumKey", fake_enum_key)
    monkeypatch.setattr(winreg, "QueryValueEx", fake_query_value_ex)

    result = gpu_probe._probe_windows_registry()

    assert result is not None
    assert result.vendor == GpuVendor.AMD
    assert result.name == "AMD Radeon RX 5700 XT"
    assert result.vram_total_mb == 8573157376 // (1024 * 1024)


@_requires_windows
def test_windows_registry_probe_skips_unclassifiable_provider(monkeypatch):
    """A ProviderName that isn't AMD/Intel-shaped (e.g. a basic software
    render adapter) must be skipped rather than guessed at."""
    monkeypatch.setattr(gpu_probe.os, "name", "nt")
    import winreg

    adapter_data = {
        "0000": {
            "ProviderName": "Microsoft Basic Render Driver",
            "DriverDesc": "Microsoft Basic Render Driver",
            "HardwareInformation.qwMemorySize": 0,
        },
    }

    def fake_open_key(key, subpath):
        if subpath in adapter_data:
            return _FakeAdapterKey(subpath)
        return _FakeAdapterKey("class")

    def fake_enum_key(key, index):
        if index == 0:
            return "0000"
        raise OSError("no more subkeys")

    def fake_query_value_ex(key, value_name):
        return adapter_data[key.name][value_name], 0

    monkeypatch.setattr(winreg, "OpenKey", fake_open_key)
    monkeypatch.setattr(winreg, "EnumKey", fake_enum_key)
    monkeypatch.setattr(winreg, "QueryValueEx", fake_query_value_ex)

    assert gpu_probe._probe_windows_registry() is None




def test_classify_provider_name_amd():
    assert gpu_probe._classify_provider_name("Advanced Micro Devices, Inc.") == GpuVendor.AMD


def test_classify_provider_name_intel():
    assert gpu_probe._classify_provider_name("Intel Corporation") == GpuVendor.INTEL


def test_classify_provider_name_unknown_returns_none():
    assert gpu_probe._classify_provider_name("Microsoft Basic Render Driver") is None


def test_classify_provider_name_non_string_returns_none():
    assert gpu_probe._classify_provider_name(None) is None




def test_probe_gpu_caches_result_across_multiple_calls(monkeypatch):
    """The underlying detection chain must run at most once per process,
    regardless of how many times/call sites invoke `probe_gpu()` -- closes
    the class of problem PR #21 iteration 2 found (~5 uncached calls per
    `check_status()` tick from independent call sites)."""
    gpu_probe.clear_cache()
    call_count = 0

    def _counting_registry_probe():
        nonlocal call_count
        call_count += 1
        return GpuProbeResult(vendor=GpuVendor.AMD, name="fake amd")

    monkeypatch.setattr(gpu_probe, "_probe_torch_cuda", lambda: None)
    monkeypatch.setattr(gpu_probe, "_probe_nvidia_smi", lambda: None)
    monkeypatch.setattr(gpu_probe, "_probe_windows_registry", _counting_registry_probe)

    first = gpu_probe.probe_gpu()
    second = gpu_probe.probe_gpu()
    third = gpu_probe.probe_gpu()

    assert call_count == 1
    assert first is second is third
    assert first.vendor == GpuVendor.AMD


@pytest.mark.asyncio
async def test_warm_gpu_probe_cache_populates_cache_so_request_path_reuses_it(monkeypatch):
    """AC 12 (Spec 028 Item 2): app.main._warm_gpu_probe_cache() -- the
    coroutine lifespan() schedules fire-and-forget at startup -- must warm
    gpu_probe's process-lifetime cache off-thread, so a later call from the
    request path (e.g. a local dictation's own device detection) reuses the
    cached result instead of spawning nvidia-smi/reading the registry again."""
    from app.main import _warm_gpu_probe_cache

    gpu_probe.clear_cache()
    monkeypatch.setenv("JUSTSAY_GPU_VENDOR", "amd")

    probe_source_calls = {"n": 0}
    real_env_override = gpu_probe._probe_env_override

    def _counting_env_override():
        probe_source_calls["n"] += 1
        return real_env_override()

    monkeypatch.setattr(gpu_probe, "_probe_env_override", _counting_env_override)

    await _warm_gpu_probe_cache()

    result = gpu_probe.probe_gpu()

    assert probe_source_calls["n"] == 1
    assert result.vendor == GpuVendor.AMD


@pytest.mark.asyncio
async def test_warm_gpu_probe_cache_swallows_probe_failure(monkeypatch):
    """A failed probe at startup must not raise -- _detect_device() simply
    pays the cost later exactly as it does today."""
    from app.main import _warm_gpu_probe_cache

    gpu_probe.clear_cache()

    def _boom():
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(gpu_probe, "probe_gpu", _boom)

    await _warm_gpu_probe_cache()


def test_lifespan_schedules_gpu_probe_warmup_task(spawn_spy):
    """Confirms app.main.lifespan() actually schedules both of its
    fire-and-forget call sites (the GPU-probe warm-up and the vector-store
    background indexer sweep) through app.core.tasks.spawn_background_task
    -- not just that the underlying coroutines work in isolation.

    Rewritten for Spec 032 (AC 7, AC 13): spies on the shared helper instead
    of monkeypatching the stdlib asyncio.create_task globally, since both
    lifespan() call sites now route through spawn_background_task() rather
    than calling asyncio.create_task() directly. Uses the shared `spawn_spy`
    fixture (conftest.py) rather than a locally-defined spy closure (GitHub
    review finding 3).
    """
    from fastapi.testclient import TestClient

    import app.main as main_module

    with TestClient(main_module.app):
        pass

    assert "gpu-probe-warmup" in spawn_spy.names
    assert "vector-store-indexer" in spawn_spy.names


def test_clear_cache_forces_a_fresh_probe_on_next_call(monkeypatch):
    """The test/dev seam: `clear_cache()` must genuinely bust the cache, not
    just be a no-op -- otherwise flipping `JUSTSAY_GPU_VENDOR` (or a mocked
    source) mid-process would never be observed."""
    gpu_probe.clear_cache()
    results = iter(
        [
            GpuProbeResult(vendor=GpuVendor.AMD, name="first"),
            GpuProbeResult(vendor=GpuVendor.INTEL, name="second"),
        ]
    )
    monkeypatch.setattr(gpu_probe, "_probe_torch_cuda", lambda: None)
    monkeypatch.setattr(gpu_probe, "_probe_nvidia_smi", lambda: None)
    monkeypatch.setattr(gpu_probe, "_probe_windows_registry", lambda: next(results))

    first = gpu_probe.probe_gpu()
    gpu_probe.clear_cache()
    second = gpu_probe.probe_gpu()

    assert first.vendor == GpuVendor.AMD
    assert second.vendor == GpuVendor.INTEL
