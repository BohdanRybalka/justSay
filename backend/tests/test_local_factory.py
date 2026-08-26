"""Tests for `app.stt.local_factory.is_macos_arm64` and `get_local_provider_class`.

Platform branches are exercised by monkeypatching `sys.platform` and
`platform.machine`. These tests call the factory directly without going
through `_get_local`, so they must opt out of the autouse
`_force_faster_whisper_for_local` fixture — done at the module level via
`pytestmark = pytest.mark.no_factory_stub`.
"""

import pytest

pytestmark = pytest.mark.no_factory_stub


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


def test_factory_returns_whisper_cpp_server_on_macos_arm64(monkeypatch):
    """On Apple Silicon the factory picks the Metal whisper.cpp provider --
    this is the routing that Local mode on macOS depends on entirely."""
    _stub_platform(monkeypatch, "darwin", "arm64")
    from app.stt.local_factory import get_local_provider_class
    from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider

    assert get_local_provider_class() is WhisperCppServerSTTProvider


def test_macos_arm64_routing_never_probes_the_gpu(monkeypatch):
    """Apple Silicon always has Metal, so the macOS branch must return before
    any `probe_gpu()` call -- the probe is uncached and expensive, and its
    `GpuVendor` enum has no `apple` member to describe the result with."""
    _stub_platform(monkeypatch, "darwin", "arm64")
    monkeypatch.setattr("os.name", "nt")

    def _explode():
        raise AssertionError("probe_gpu() must not run on the macOS arm64 path")

    monkeypatch.setattr("app.core.gpu_probe.probe_gpu", _explode)
    from app.stt.local_factory import LocalProviderKind, get_local_provider_kind

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER




def test_kind_is_whisper_cpp_server_on_macos_arm64_regardless_of_os_name_or_vendor(monkeypatch):
    """macOS arm64 wins outright — `os.name`/vendor are irrelevant once
    `is_macos_arm64()` is true."""
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import LocalProviderKind, get_local_provider_kind

    _stub_platform(monkeypatch, "darwin", "arm64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor.AMD)

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER


def test_kind_is_whisper_cpp_server_on_windows_amd(monkeypatch):
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )
    from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider

    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor.AMD)

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER
    assert get_local_provider_class() is WhisperCppServerSTTProvider


def test_kind_is_whisper_cpp_server_on_windows_intel(monkeypatch):
    from app.core.gpu_probe import GpuVendor
    from app.stt.local_factory import (
        LocalProviderKind,
        get_local_provider_class,
        get_local_provider_kind,
    )
    from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider

    _stub_platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr("os.name", "nt")
    _stub_vendor(monkeypatch, GpuVendor.INTEL)

    assert get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER
    assert get_local_provider_class() is WhisperCppServerSTTProvider


@pytest.mark.parametrize("vendor_name", ["NVIDIA", "NONE"])
def test_kind_is_faster_whisper_on_windows_nvidia_or_none(monkeypatch, vendor_name):
    """Unchanged — explicit regression test: Windows + NVIDIA/no GPU never
    routes onto the whisper.cpp-server path."""
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
    """Importing the factory must not pull in faster_whisper or httpx-backed
    provider modules.

    Catches accidental top-level `import faster_whisper` regressions that
    would crash on platforms missing the package.
    """
    from tests.conftest import assert_module_binds_no_third_party

    assert_module_binds_no_third_party(
        "app.stt.local_factory", ("faster_whisper", "local_whisper_cpp")
    )
