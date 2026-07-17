"""Tests for `app.stt.local_factory.is_macos_arm64` and `get_local_provider_class`.

Platform branches are exercised by monkeypatching `sys.platform` and
`platform.machine`. These tests call the factory directly without going
through `_get_local`, so they must opt out of the autouse
`_force_faster_whisper_for_local` fixture — done at the module level via
`pytestmark = pytest.mark.mlx`.
"""

import pytest

pytestmark = pytest.mark.mlx


def _stub_platform(monkeypatch, sys_platform: str, machine: str) -> None:
    """Pin `sys.platform` and `platform.machine()` to the given values."""
    monkeypatch.setattr("sys.platform", sys_platform)
    monkeypatch.setattr("platform.machine", lambda: machine)


def _stub_vendor(monkeypatch, vendor) -> None:
    """Pin `app.core.gpu_probe.probe_gpu()`'s reported vendor."""
    from app.core.gpu_probe import GpuProbeResult

    monkeypatch.setattr("app.core.gpu_probe.probe_gpu", lambda: GpuProbeResult(vendor=vendor))


def test_is_macos_arm64_true_on_darwin_arm64(monkeypatch):
    _stub_platform(monkeypatch, "darwin", "arm64")
    from app.stt.local_factory import is_macos_arm64

    assert is_macos_arm64() is True


def test_is_macos_arm64_false_on_darwin_rosetta(monkeypatch):
    """Rosetta x86 Python on Apple Silicon reports machine() == 'x86_64'."""
    _stub_platform(monkeypatch, "darwin", "x86_64")
    from app.stt.local_factory import is_macos_arm64

    assert is_macos_arm64() is False


def test_is_macos_arm64_false_on_windows(monkeypatch):
    _stub_platform(monkeypatch, "win32", "AMD64")
    from app.stt.local_factory import is_macos_arm64

    assert is_macos_arm64() is False


def test_is_macos_arm64_false_on_linux(monkeypatch):
    _stub_platform(monkeypatch, "linux", "x86_64")
    from app.stt.local_factory import is_macos_arm64

    assert is_macos_arm64() is False


def test_factory_returns_local_on_windows(monkeypatch):
    """Windows + no AMD/Intel GPU (spec 018 didn't exist when this test was
    first written — vendor now matters on Windows) still falls back to
    faster-whisper. `os.name`/vendor are pinned explicitly (not just
    `sys.platform`) so this stays deterministic on a machine with a real
    AMD/Intel GPU — this project's own dev box, since spec 018.
    """
    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    from app.core.gpu_probe import GpuProbeResult, GpuVendor
    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu", lambda: GpuProbeResult(vendor=GpuVendor.NONE)
    )
    from app.stt.local import LocalSTTProvider
    from app.stt.local_factory import get_local_provider_class

    assert get_local_provider_class() is LocalSTTProvider


def test_factory_returns_local_on_linux(monkeypatch):
    """`os.name` is pinned explicitly (not just `sys.platform`) — spec 018's
    Windows+AMD/Intel routing checks the real `os.name`, which a bare
    `sys.platform` stub does not affect.
    """
    _stub_platform(monkeypatch, "linux", "x86_64")
    monkeypatch.setattr("os.name", "posix")
    from app.stt.local import LocalSTTProvider
    from app.stt.local_factory import get_local_provider_class

    assert get_local_provider_class() is LocalSTTProvider


def test_factory_returns_local_on_macos_intel(monkeypatch):
    """macOS Intel falls back to faster-whisper CPU path. `os.name` pinned
    explicitly — see `test_factory_returns_local_on_linux`'s docstring.
    """
    _stub_platform(monkeypatch, "darwin", "x86_64")
    monkeypatch.setattr("os.name", "posix")
    from app.stt.local import LocalSTTProvider
    from app.stt.local_factory import get_local_provider_class

    assert get_local_provider_class() is LocalSTTProvider


def test_factory_returns_mlx_on_macos_arm64(monkeypatch):
    """On Apple Silicon the factory picks the MLX provider."""
    _stub_platform(monkeypatch, "darwin", "arm64")
    from app.stt.local_factory import get_local_provider_class

    cls = get_local_provider_class()
    assert cls.__name__ == "MLXWhisperSTTProvider"
    # Asserting on class name (not import) keeps this test green even on a
    # Windows dev box where `mlx_whisper` is not installed — the MLX provider
    # module imports `huggingface_hub` at module top, which IS installed.


# --- get_local_provider_kind() branch matrix (spec 018) ---


def test_kind_is_apple_mlx_on_macos_arm64_regardless_of_os_name_or_vendor(monkeypatch):
    """macOS arm64 wins outright — `os.name`/vendor are irrelevant once
    `is_macos_arm64()` is true."""
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import LocalProviderKind, get_local_provider_kind

    _stub_platform(monkeypatch, "darwin", "arm64")
    monkeypatch.setattr("os.name", "nt")  # deliberately contradictory — must still be ignored
    _stub_vendor(monkeypatch, GpuVendor.AMD)

    assert get_local_provider_kind() is LocalProviderKind.APPLE_MLX


def test_kind_is_vulkan_on_windows_amd(monkeypatch):
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )
    from app.stt.local_vulkan import WhisperCppVulkanSTTProvider

    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor.AMD)

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_VULKAN
    assert get_local_provider_class() is WhisperCppVulkanSTTProvider


def test_kind_is_vulkan_on_windows_intel(monkeypatch):
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )
    from app.stt.local_vulkan import WhisperCppVulkanSTTProvider

    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor.INTEL)

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_VULKAN
    assert get_local_provider_class() is WhisperCppVulkanSTTProvider


@pytest.mark.parametrize("vendor_name", ["NVIDIA", "NONE"])
def test_kind_is_faster_whisper_on_windows_nvidia_or_none(monkeypatch, vendor_name):
    """Unchanged — explicit regression test: Windows + NVIDIA/no GPU never
    routes onto the Vulkan path."""
    from app.core.gpu_probe import GpuVendor
    from app.stt.local import LocalSTTProvider
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )

    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor[vendor_name])

    assert get_local_provider_kind() is LocalProviderKind.FASTER_WHISPER
    assert get_local_provider_class() is LocalSTTProvider


@pytest.mark.parametrize("vendor_name", ["AMD", "INTEL"])
def test_kind_is_faster_whisper_on_non_windows_amd_or_intel(monkeypatch, vendor_name):
    """AMD/Intel Vulkan routing is Windows-only — Linux/macOS-Intel are not
    supported Local-mode target platforms per CLAUDE.md, and `release.yml`'s
    build matrix has no leg for either."""
    from app.core.gpu_probe import GpuVendor
    from app.stt.local import LocalSTTProvider
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )

    _stub_platform(monkeypatch, "linux", "x86_64")
    monkeypatch.setattr("os.name", "posix")
    _stub_vendor(monkeypatch, GpuVendor[vendor_name])

    assert get_local_provider_kind() is LocalProviderKind.FASTER_WHISPER
    assert get_local_provider_class() is LocalSTTProvider


def test_factory_module_imports_no_third_party_at_module_level():
    """Importing the factory must not pull in mlx_whisper or faster_whisper.

    Catches accidental top-level `import mlx_whisper` regressions that would
    crash on platforms missing the package.
    """
    import importlib
    import sys

    # Drop cached entries first so the re-import actually runs module-body code.
    for name in list(sys.modules):
        if name == "app.stt.local_factory":
            del sys.modules[name]
    importlib.import_module("app.stt.local_factory")
    # Plain factory import must not have imported the heavy deps.
    # (They may already be in sys.modules from earlier tests — we only
    # assert the factory module itself doesn't reference them at top level.)
    factory_mod = sys.modules["app.stt.local_factory"]
    assert not hasattr(factory_mod, "mlx_whisper")
    assert not hasattr(factory_mod, "faster_whisper")
