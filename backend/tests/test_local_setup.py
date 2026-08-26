import asyncio
import logging
import pathlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.core.utils import sse_event
from app.stt import local_setup
from app.stt.config import STTSettings
from app.stt.local_factory import get_local_provider_class as _real_get_local_provider_class
from app.stt.local_factory import get_local_provider_kind as _real_get_local_provider_kind
from app.stt.local_setup import (
    LocalSttStatus,
    _check_package_installed,
    _detect_gpu,
    check_status,
    install_local_packages,
)


def _patches(installed: bool, gpu: tuple[bool, str | None, str]):
    """Standard patch set: stub package detection + GPU detection.

    `gpu` is `(available, name, vendor)` — the `_detect_gpu()` 3-tuple.
    """
    return [
        patch.object(local_setup, "_check_package_installed", return_value=installed),
        patch.object(local_setup, "_detect_gpu", return_value=gpu),
    ]


def _apply(patches):
    """Enter a list of patches as a single context using ExitStack."""
    from contextlib import ExitStack

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def test_check_status_reports_installed_package():
    settings = STTSettings(whisper_model_size="large-v3-turbo", whisper_device="auto")

    with _apply(_patches(True, (False, None, "none"))):
        status = check_status(settings)

    assert isinstance(status, LocalSttStatus)
    assert status.package_installed is True
    assert status.model_loaded is False
    assert status.last_error is None
    assert status.model_ram_mb is None
    assert status.model_name == "large-v3-turbo"
    assert status.gpu_available is False
    assert status.gpu_name is None
    assert status.gpu_vendor == "none"
    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_uses_cuda_when_gpu_auto():
    settings = STTSettings(whisper_device="auto")

    with _apply(_patches(True, (True, "NVIDIA GeForce RTX 3060", "nvidia"))):
        status = check_status(settings)

    assert status.gpu_available is True
    assert status.gpu_name == "NVIDIA GeForce RTX 3060"
    assert status.gpu_vendor == "nvidia"
    assert status.device == "cuda"
    assert status.compute_type == "float16"


def test_check_status_respects_explicit_cpu_device():
    settings = STTSettings(whisper_device="cpu")

    with _apply(_patches(True, (True, "RTX 3060", "nvidia"))):
        status = check_status(settings)

    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_reports_missing_package():
    settings = STTSettings()

    with _apply(_patches(False, (False, None, "none"))):
        status = check_status(settings)

    assert status.package_installed is False
    assert status.model_loaded is False


def test_check_status_reports_amd_gpu_name_and_vendor_but_not_available():
    """AMD is detected (name + vendor populated) but gpu_available stays False
    — faster-whisper has no AMD backend (spec 014)."""
    settings = STTSettings(whisper_device="auto")

    with _apply(_patches(True, (False, "AMD Radeon RX 5700 XT", "amd"))):
        status = check_status(settings)

    assert status.gpu_available is False
    assert status.gpu_vendor == "amd"
    assert status.gpu_name == "AMD Radeon RX 5700 XT"
    assert status.device == "cpu"


def test_check_status_surfaces_last_load_error():
    """When _get_model latched an error, status.last_error must contain it."""
    from app.stt import clear_cache as clear_stt_cache

    clear_stt_cache()
    settings = STTSettings()
    from app.stt import _get_local

    provider = _get_local(settings)
    provider._last_load_error = "OSError: [WinError 126] DLL not found"
    try:
        with _apply(_patches(True, (False, None, "none"))):
            status = check_status(settings)
        assert status.last_error == "OSError: [WinError 126] DLL not found"
        assert status.model_loaded is False
    finally:
        clear_stt_cache()


def test_get_local_load_error_returns_none_before_provider_instantiation():
    from app.stt import clear_cache as clear_stt_cache
    from app.stt import get_local_load_error

    clear_stt_cache()
    settings = STTSettings()
    assert get_local_load_error(settings) is None




def test_check_package_installed_true(monkeypatch):
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        assert _check_package_installed() is True


def test_check_package_installed_false(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _check_package_installed() is False




def test_detect_gpu_returns_false_when_probe_reports_none(monkeypatch):
    """`_detect_gpu` delegates to `gpu_probe.probe_gpu()` (spec 014); when the
    probe finds nothing, availability/name/vendor are all "empty"."""
    from app.core.gpu_probe import GpuProbeResult, GpuVendor

    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(vendor=GpuVendor.NONE),
    )
    available, name, vendor = _detect_gpu()
    assert available is False
    assert name is None
    assert vendor == "none"


def test_detect_gpu_returns_name_when_cuda_available(monkeypatch):
    from app.core.gpu_probe import GpuProbeResult, GpuVendor

    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(vendor=GpuVendor.NVIDIA, name="RTX 4090"),
    )
    available, name, vendor = _detect_gpu()

    assert available is True
    assert name == "RTX 4090"
    assert vendor == "nvidia"


def test_detect_gpu_reports_amd_name_and_vendor_but_not_available(monkeypatch):
    """AMD is detected (name + vendor populated) but `available` stays False —
    faster-whisper (CTranslate2) has no AMD backend (spec 014)."""
    from app.core.gpu_probe import GpuProbeResult, GpuVendor

    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(vendor=GpuVendor.AMD, name="AMD Radeon RX 5700 XT"),
    )
    available, name, vendor = _detect_gpu()

    assert available is False
    assert name == "AMD Radeon RX 5700 XT"
    assert vendor == "amd"




def test_sse_format_is_event_plus_data():
    out = sse_event("progress", {"status": "downloading"})
    assert out.startswith("event: progress\n")
    assert 'data: {"status": "downloading"}' in out
    assert out.endswith("\n\n")




@pytest.mark.asyncio
async def test_install_skipped_when_already_installed():
    with patch.object(local_setup, "_check_package_installed", return_value=True):
        events = [e async for e in install_local_packages()]

    assert len(events) == 1
    assert "already_installed" in events[0]


@pytest.mark.asyncio
async def test_install_emits_done_on_success():
    with patch.object(local_setup, "_check_package_installed", return_value=False), patch.object(
        local_setup, "_run_pip_install", return_value=(0, "ok")
    ):
        events = [e async for e in install_local_packages()]

    assert any("progress" in e for e in events)
    assert any("success" in e for e in events)


@pytest.mark.asyncio
async def test_install_emits_error_on_failure():
    with patch.object(local_setup, "_check_package_installed", return_value=False), patch.object(
        local_setup, "_run_pip_install", return_value=(1, "pip: something went wrong")
    ):
        events = [e async for e in install_local_packages()]

    assert any("event: error" in e for e in events)


@pytest.mark.asyncio
async def test_install_concurrent_locked():
    """When the install lock is already held, a second caller gets an error SSE."""
    await local_setup._install_lock.acquire()
    try:
        events = [e async for e in install_local_packages()]
    finally:
        local_setup._install_lock.release()

    assert len(events) == 1
    assert "event: error" in events[0]
    assert "already in progress" in events[0]


@pytest.mark.asyncio
async def test_install_refused_in_frozen_binary(monkeypatch):
    """In a PyInstaller-frozen sidecar `sys.frozen=True` and pip install is meaningless.

    The endpoint must surface a clear error SSE rather than running pip from
    the random `_MEIPASS` cwd that `Path(__file__).resolve()` produces.
    """
    monkeypatch.setattr(local_setup.sys, "frozen", True, raising=False)

    events = [e async for e in install_local_packages()]

    assert len(events) == 1
    assert "event: error" in events[0]
    assert "not supported in the packaged build" in events[0]




def test_run_pip_install_uses_the_local_extras_on_every_platform(monkeypatch):
    """With the Apple-Silicon Python package gone there is no macOS pip path
    left: the accelerated platforms resolve a bundled binary and never reach
    `_run_pip_install`, so one extras name covers everything that does."""
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(local_setup.subprocess, "run", _fake_run)
    code, _ = local_setup._run_pip_install()
    assert code == 0
    assert ".[local]" in captured["cmd"]
    assert sum(part.startswith(".[") for part in captured["cmd"]) == 1


def test_check_package_installed_resolves_the_binary_on_macos_arm64(monkeypatch):
    """macOS arm64 now checks for the whisper-server binary, exactly as
    Windows AMD/Intel does -- there is no package to import any more."""
    _stub_whisper_cpp_server_kind(monkeypatch)
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    monkeypatch.setattr(
        "app.stt.local_whisper_cpp_cmd.resolve_binary_path",
        lambda: pathlib.Path("/opt/whisper-cpp-metal/whisper-server"),
    )

    assert _check_package_installed() is True


def test_check_package_installed_false_when_the_binary_is_missing_on_macos_arm64(monkeypatch):
    _stub_whisper_cpp_server_kind(monkeypatch)
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    monkeypatch.setattr("app.stt.local_whisper_cpp_cmd.resolve_binary_path", lambda: None)

    assert _check_package_installed() is False


def test_check_package_installed_never_imports_the_removed_apple_package(monkeypatch):
    """Spec 068's core regression. The removed Apple-Silicon Python package
    was never installed into the shipped macOS sidecar, so importing it here
    always failed and Local mode read as permanently broken on every Mac. An
    import of it reappearing on this path resurrects that outage.

    The package name is assembled rather than written out because the spec's
    own acceptance grep forbids the literal anywhere under `backend/` -- same
    idiom as `test_local_whisper_cpp_cmd.py`'s `"shell" + "=" + "True"`.
    """
    import builtins

    removed_package = "m" + "lx_whisper"

    _stub_whisper_cpp_server_kind(monkeypatch)
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    monkeypatch.setattr(
        "app.stt.local_whisper_cpp_cmd.resolve_binary_path",
        lambda: pathlib.Path("/opt/whisper-cpp-metal/whisper-server"),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == removed_package:
            raise AssertionError(f"{name} must never be imported again")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _check_package_installed() is True


def test_detect_gpu_reports_apple_silicon_on_macos_arm64(monkeypatch):
    """On M1+ we must return the Metal label without importing torch.

    Side-bonus: confirms _detect_gpu's macOS branch runs **before** any
    `import torch` call.
    """
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch must not be imported on macOS arm64 path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    available, name, vendor = _detect_gpu()
    assert available is True
    assert name == "Apple Silicon (Metal)"
    assert vendor == "apple"


def test_check_status_macos_arm64_reports_metal_device_and_float16(monkeypatch):
    """On macOS arm64 `check_status` must return device='metal',
    compute_type='float16'."""
    settings = STTSettings(whisper_model_size="large-v3-turbo")
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)

    with _apply(_patches(True, (True, "Apple Silicon (Metal)", "apple"))):
        status = check_status(settings)

    assert status.device == "metal"
    assert status.compute_type == "float16"
    assert status.gpu_available is True
    assert status.gpu_name == "Apple Silicon (Metal)"
    assert status.gpu_vendor == "apple"


def test_check_status_macos_arm64_never_constructs_a_gpu_vendor_from_apple(monkeypatch):
    """`_detect_gpu()` returns the vendor string "apple", which is NOT a
    member of `GpuVendor` (nvidia/amd/intel/none). `check_status()`'s macOS
    branch must return before the `else` branch's `GpuVendor(gpu_vendor)`
    call, or every Mac gets a 500 from `GET /stt/local/status` instead of a
    status object. Removing that early branch looks like a simplification and
    is a total outage on one platform.
    """
    from app.core import gpu_probe

    settings = STTSettings(whisper_model_size="large-v3-turbo")
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)

    real_gpu_vendor = gpu_probe.GpuVendor

    def _exploding_gpu_vendor(value):
        raise AssertionError(
            f"check_status() constructed GpuVendor({value!r}) on macOS arm64 -- "
            "the early is_macos_arm64() branch was lost"
        )

    monkeypatch.setattr(gpu_probe, "GpuVendor", _exploding_gpu_vendor)
    try:
        with _apply(_patches(True, (True, "Apple Silicon (Metal)", "apple"))):
            status = check_status(settings)
    finally:
        monkeypatch.setattr(gpu_probe, "GpuVendor", real_gpu_vendor)

    assert status.device == "metal"


def test_gpu_vendor_enum_still_has_no_apple_member():
    """Pins the premise of the test above. If `apple` is ever added to
    `GpuVendor`, that regression test stops proving anything and should be
    rewritten rather than silently kept."""
    from app.core.gpu_probe import GpuVendor

    assert {member.value for member in GpuVendor} == {"nvidia", "amd", "intel", "none"}





def _stub_whisper_cpp_server_kind(monkeypatch) -> None:
    from app.stt.local_factory import LocalProviderKind

    monkeypatch.setattr(
        local_setup,
        "get_local_provider_kind",
        lambda *args, **kwargs: LocalProviderKind.WHISPER_CPP_SERVER,
    )


def test_check_status_whisper_cpp_server_kind_reports_device_and_compute_type(monkeypatch):
    _stub_whisper_cpp_server_kind(monkeypatch)
    settings = STTSettings(whisper_model_size="large-v3-turbo")

    with _apply(_patches(True, (False, "AMD Radeon RX 5700 XT", "amd"))):
        status = check_status(settings)

    assert status.device == "vulkan"
    assert status.compute_type == "float16"


def test_check_status_whisper_cpp_server_kind_reports_gpu_available_true(monkeypatch):
    """Regression for Stage 3 review issue #2: a Vulkan-accelerated
    AMD/Intel session must not report the contradictory
    device: "vulkan" + gpu_available: false pair. `_detect_gpu()` itself
    still returns its NVIDIA-only `available` bool (False for AMD) — the
    status object's `gpu_available` is derived from the final `device`
    instead."""
    _stub_whisper_cpp_server_kind(monkeypatch)
    settings = STTSettings(whisper_model_size="large-v3-turbo")

    with _apply(_patches(True, (False, "AMD Radeon RX 5700 XT", "amd"))):
        status = check_status(settings)

    assert status.device == "vulkan"
    assert status.gpu_available is True
    assert status.gpu_vendor == "amd"


def test_check_status_probes_gpu_at_most_once_through_the_real_unmocked_provider_class_path(
    monkeypatch,
):
    """Regression for the GitHub review on PR #21, iteration 2: the
    iteration-1 fix (an optional `vendor` param threaded through
    `check_status()`'s own `get_local_provider_kind()` call) only closed ONE
    of several call sites that independently reach `get_local_provider_kind()`
    with no vendor -- `get_local_provider_class()` (`local_factory.py:87`),
    reached via `_check_package_installed()`'s own WHISPER_CPP_SERVER branch
    check, `is_model_loaded()` (called *twice* inside `check_status()`: once
    for `model_loaded=`, once more inside the `model_ram_mb=... if
    is_model_loaded() else None` ternary), and `get_local_load_error()`.

    The prior version of this test (before this fix) only proved the
    call-count reduction held for `_detect_gpu()`'s single direct call,
    because it (a) stubbed `_check_package_installed()` out entirely --
    bypassing its own internal `get_local_provider_kind()` call site -- and
    (b) never touched `local_factory.get_local_provider_class`, which the
    autouse `_force_faster_whisper_for_local` fixture patches to a lambda
    that never calls `get_local_provider_kind()` at all. That made the old
    test pass for a reason that had nothing to do with the fix: on GitHub
    review, PR #21 iteration 2, confirmed those other call sites still ran
    the real, uncached `probe_gpu()` ~5 times per `check_status()` tick in
    production.

    Fixed at the source instead of threading `vendor` through every call
    site: `app.core.gpu_probe.probe_gpu()` now caches its result for the
    process lifetime, so it no longer matters how many independent,
    uncoordinated call sites reach it -- the underlying detection source
    only ever runs once. This test restores BOTH
    `local_factory.get_local_provider_class` and
    `local_setup.get_local_provider_kind` to their real, unpatched
    implementations (undoing the autouse fixture for this one test) and
    counts calls to the underlying probe *source*
    (`gpu_probe._probe_env_override`), not to `probe_gpu()` itself -- so the
    assertion holds regardless of which higher-level function reaches it,
    proving the property against the real call graph rather than a mock
    that would trivially always read 1. `JUSTSAY_GPU_VENDOR=amd` makes the
    routing outcome deterministic on any machine (real hardware is not
    involved), matching this module's own documented test-seam intent.
    """
    from pathlib import Path

    from app.core import gpu_probe

    gpu_probe.clear_cache()
    monkeypatch.setenv("JUSTSAY_GPU_VENDOR", "amd")

    from app.stt import local_factory

    monkeypatch.setattr(local_factory, "get_local_provider_class", _real_get_local_provider_class)
    monkeypatch.setattr(local_setup, "get_local_provider_kind", _real_get_local_provider_kind)
    monkeypatch.setattr(local_setup, "is_macos_arm64", lambda: False)
    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr(
        "app.stt.local_whisper_cpp_cmd.resolve_binary_path", lambda: Path("whisper-server.exe")
    )

    probe_source_calls = {"n": 0}
    real_env_override = gpu_probe._probe_env_override

    def _counting_env_override():
        probe_source_calls["n"] += 1
        return real_env_override()

    monkeypatch.setattr(gpu_probe, "_probe_env_override", _counting_env_override)

    settings = STTSettings(whisper_model_size="large-v3-turbo")
    status = check_status(settings)

    assert probe_source_calls["n"] == 1
    assert status.device == "vulkan"
    assert status.gpu_available is True
    assert status.gpu_vendor == "amd"


def test_check_package_installed_vulkan_kind_true_when_binary_resolves(monkeypatch):
    _stub_whisper_cpp_server_kind(monkeypatch)
    from pathlib import Path

    monkeypatch.setattr(
        "app.stt.local_whisper_cpp_cmd.resolve_binary_path", lambda: Path("whisper-server.exe")
    )
    assert _check_package_installed() is True


def test_check_package_installed_vulkan_kind_false_when_binary_missing(monkeypatch):
    _stub_whisper_cpp_server_kind(monkeypatch)
    monkeypatch.setattr("app.stt.local_whisper_cpp_cmd.resolve_binary_path", lambda: None)
    assert _check_package_installed() is False


def test_estimate_model_ram_mb_returns_none_for_vulkan_kind(monkeypatch):
    """The model lives in the separate whisper-server child process's own
    address space — reporting this (FastAPI backend) process's RSS would be
    actively misleading, not just imprecise."""
    _stub_whisper_cpp_server_kind(monkeypatch)
    assert local_setup._estimate_model_ram_mb() is None


@pytest.mark.asyncio
async def test_ensure_local_ready_vulkan_kind_skips_pip_install_when_binary_present(monkeypatch):
    _stub_whisper_cpp_server_kind(monkeypatch)
    provider = _FakePrewarmProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    def _boom():
        raise AssertionError("_run_pip_install must not be called for the Vulkan kind")

    monkeypatch.setattr(local_setup, "_run_pip_install", _boom)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 1
    assert local_setup._prewarm_error is None


@pytest.mark.asyncio
async def test_ensure_local_ready_vulkan_kind_sets_actionable_error_when_binary_missing(
    monkeypatch,
):
    _stub_whisper_cpp_server_kind(monkeypatch)
    provider = _FakePrewarmProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: False)

    def _boom():
        raise AssertionError("_run_pip_install must not be called for the Vulkan kind")

    monkeypatch.setattr(local_setup, "_run_pip_install", _boom)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 0
    assert local_setup._prewarm_error is not None
    assert "whisper-server binary not found" in local_setup._prewarm_error




def test_check_status_surfaces_prewarm_error_when_no_provider_level_error():
    from app.stt import clear_cache as clear_stt_cache

    clear_stt_cache()
    settings = STTSettings()
    local_setup._prewarm_error = "pip install failed: boom"
    try:
        with _apply(_patches(True, (False, None, "none"))):
            status = check_status(settings)
        assert status.last_error == "pip install failed: boom"
    finally:
        local_setup._prewarm_error = None
        clear_stt_cache()


def test_check_status_prefers_provider_error_over_prewarm_error():
    from app.stt import _get_local
    from app.stt import clear_cache as clear_stt_cache

    clear_stt_cache()
    settings = STTSettings()
    provider = _get_local(settings)
    provider._last_load_error = "provider-level error"
    local_setup._prewarm_error = "prewarm-level error"
    try:
        with _apply(_patches(True, (False, None, "none"))):
            status = check_status(settings)
        assert status.last_error == "provider-level error"
    finally:
        local_setup._prewarm_error = None
        clear_stt_cache()


def test_check_status_merge_is_deterministic_when_package_missing_and_provider_error_latched():
    """Tech-debt closure (spec 015 Review iteration 1, YELLOW #3): the
    check_status() comment "mutually exclusive in practice" assumes
    _check_package_installed() never flips True -> False mid-process, so an
    install failure (_prewarm_error) and a provider load failure never
    coexist. Force that "impossible" combined state anyway and pin the merge
    outcome: get_local_load_error() or _prewarm_error -> the provider error
    always wins, regardless of package_installed being False."""
    from app.stt import _get_local
    from app.stt import clear_cache as clear_stt_cache

    clear_stt_cache()
    settings = STTSettings()
    provider = _get_local(settings)
    provider._last_load_error = "provider-level load error"
    local_setup._prewarm_error = "install-level prewarm error"
    try:
        with _apply(_patches(False, (False, None, "none"))):
            status = check_status(settings)
        assert status.last_error == "provider-level load error"
        assert status.package_installed is False
        assert status.model_loaded is False
    finally:
        local_setup._prewarm_error = None
        clear_stt_cache()




class _FakePrewarmProvider:
    """Minimal stand-in for LocalSTTProvider — tracks calls, no real model load."""

    def __init__(self, get_model=None):
        self.is_loaded = False
        self.get_model_calls = 0
        self.cleanup_calls = 0
        self._get_model_impl = get_model

    def _get_model(self):
        self.get_model_calls += 1
        if self._get_model_impl is not None:
            self._get_model_impl(self)
        else:
            self.is_loaded = True

    def cleanup(self):
        self.cleanup_calls += 1


@pytest.fixture(autouse=True)
def _reset_prewarm_state():
    """Every test in this module runs against a clean `_prewarm_error` latch,
    regardless of what backend/tests/conftest.py's autouse fixture does.

    `_prewarm_lock` used to be rebound here too, which made this module the one
    place in the suite safe from the closed-event-loop trap described in
    `conftest._reset_event_loop_bound_locks`. That is exactly how JS-110 hid:
    the barrier test in `test_pipeline.py` contends the same singleton and had
    no such reset. The reset is suite-wide now and this module inherits it.

    `_active_load` (Stage 5 GitHub review, PR #34, finding 1) holds an
    `asyncio.Task` -- the same closed-event-loop hazard applies, so it is
    reset to `None` here too, before each test.
    """
    local_setup._prewarm_error = None
    local_setup._active_load = None
    yield
    local_setup._prewarm_error = None


@pytest.mark.prewarm
def test_maybe_prewarm_local_is_noop_for_cloud_mode(monkeypatch):
    """`maybe_prewarm_local` must never schedule a task when mode isn't LOCAL.

    Also proves the early-return happens before `asyncio.create_task` would
    even be reached — calling it from a plain sync test (no running event
    loop) would itself raise if that guard were missing.

    Marked `@pytest.mark.prewarm`: without it, backend/tests/conftest.py's
    autouse fixture would replace `maybe_prewarm_local` itself with a no-op,
    making this test pass vacuously instead of exercising the real function.
    """
    called = {"n": 0}

    async def _spy(stt_settings):
        called["n"] += 1

    monkeypatch.setattr(local_setup, "ensure_local_ready", _spy)
    settings = STTSettings(mode=ProviderMode.CLOUD)
    local_setup.maybe_prewarm_local(settings)
    assert called["n"] == 0


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_maybe_prewarm_local_schedules_ensure_local_ready_for_local_mode(monkeypatch):
    """Marked `@pytest.mark.prewarm` for the same reason as the cloud-mode
    no-op test above — needs the real `maybe_prewarm_local`, not the
    autouse-patched no-op."""
    called = {"n": 0}

    async def _spy(stt_settings):
        called["n"] += 1

    monkeypatch.setattr(local_setup, "ensure_local_ready", _spy)
    settings = STTSettings(mode=ProviderMode.LOCAL)
    local_setup.maybe_prewarm_local(settings)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending)
    assert called["n"] == 1


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_maybe_prewarm_local_routes_through_spawn_background_task(
    monkeypatch, spawn_spy
):
    """AC 8 (Spec 032): maybe_prewarm_local() must schedule its
    ensure_local_ready() call through app.core.tasks.spawn_background_task()
    -- exactly once for LOCAL mode, and not at all for CLOUD mode.

    Marked @pytest.mark.prewarm for the same reason as the tests above --
    needs the real maybe_prewarm_local, not the autouse-patched no-op.

    Uses the shared `spawn_spy` fixture (conftest.py) and awaits
    `spawn_spy.tasks` -- the exact task(s) it spawned -- rather than every
    task pending on the loop (GitHub review finding 1): the old
    `asyncio.all_tasks()` sweep would also await unrelated pending tasks,
    surfacing any failure among them as an unrelated-looking error here.
    """

    async def _fake_ensure_local_ready(stt_settings):
        return None

    monkeypatch.setattr(local_setup, "ensure_local_ready", _fake_ensure_local_ready)

    local_setup.maybe_prewarm_local(STTSettings(mode=ProviderMode.CLOUD))
    assert spawn_spy.names == []

    local_setup.maybe_prewarm_local(STTSettings(mode=ProviderMode.LOCAL))
    await asyncio.gather(*spawn_spy.tasks)

    assert spawn_spy.names == ["local-stt-prewarm"]


@pytest.mark.asyncio
async def test_ensure_local_ready_noop_when_mode_not_local_at_entry(monkeypatch):
    """Never touches the provider cache when mode isn't LOCAL at entry."""

    def _boom(mode, stt_settings):
        raise AssertionError("get_provider must not be called")

    monkeypatch.setattr("app.stt.get_provider", _boom)
    settings = STTSettings(mode=ProviderMode.CLOUD)
    await local_setup.ensure_local_ready(settings)


@pytest.mark.asyncio
async def test_ensure_local_ready_fast_path_when_already_loaded(monkeypatch):
    """No install/load attempted once `provider.is_loaded` is already True."""
    provider = _FakePrewarmProvider()
    provider.is_loaded = True
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)

    def _boom():
        raise AssertionError("_check_package_installed must not be called")

    monkeypatch.setattr(local_setup, "_check_package_installed", _boom)
    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)
    assert provider.get_model_calls == 0


@pytest.mark.asyncio
async def test_ensure_local_ready_installs_then_loads_on_success(monkeypatch):
    provider = _FakePrewarmProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: False)
    monkeypatch.setattr(local_setup, "_run_pip_install", lambda: (0, "ok"))

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 1
    assert provider.is_loaded is True
    assert local_setup._prewarm_error is None


@pytest.mark.asyncio
async def test_ensure_local_ready_sets_prewarm_error_on_install_failure_and_skips_load(monkeypatch):
    provider = _FakePrewarmProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: False)
    monkeypatch.setattr(local_setup, "_run_pip_install", lambda: (1, "pip: something went wrong"))

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 0
    assert local_setup._prewarm_error == "pip: something went wrong"


@pytest.mark.asyncio
async def test_ensure_local_ready_aborts_load_if_mode_changes_during_install(monkeypatch):
    """A genuine Local -> Cloud switch that happens while the (possibly
    multi-minute) install is running must skip the load attempt entirely.

    In production a mode switch goes through `clear_cache()`, evicting the
    captured provider — simulated here via a shared cache slot so it's the
    identity check (peek_local_provider() is provider), not a mode check,
    that aborts the load (spec 015, RED-1)."""
    provider = _FakePrewarmProvider()
    cache = {"current": provider}
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: cache["current"])
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: cache["current"])
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: False)

    settings = STTSettings(mode=ProviderMode.LOCAL)

    def _install_and_flip_mode():
        settings.mode = ProviderMode.CLOUD
        cache["current"] = None
        return (0, "ok")

    monkeypatch.setattr(local_setup, "_run_pip_install", _install_and_flip_mode)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 0


@pytest.mark.asyncio
async def test_ensure_local_ready_cleans_up_orphan_after_mode_change_mid_load(monkeypatch):
    """If the mode changes away from LOCAL while `_get_model()` itself is in
    flight, `cleanup()` must run once the load settles.

    A genuine mode switch goes through `clear_cache()` in production, so the
    fake `_get_model` here evicts the provider from a shared cache slot too —
    it's the identity check, not the mode flip itself, that must trigger
    cleanup (spec 015, RED-1)."""
    settings = STTSettings(mode=ProviderMode.LOCAL)
    cache = {"current": None}

    def _flip_mode_during_load(self_provider):
        settings.mode = ProviderMode.CLOUD
        cache["current"] = None
        self_provider.is_loaded = True

    provider = _FakePrewarmProvider(get_model=_flip_mode_during_load)
    cache["current"] = provider
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: cache["current"])
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: cache["current"])
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 1
    assert provider.cleanup_calls == 1


@pytest.mark.asyncio
async def test_ensure_local_ready_swallows_get_model_exception_without_cleanup(monkeypatch):
    """A load failure is swallowed (the provider already latched its own
    error) and must not trigger cleanup() while the mode is still LOCAL."""

    def _raise(self_provider):
        raise RuntimeError("boom")

    provider = _FakePrewarmProvider(get_model=_raise)
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await local_setup.ensure_local_ready(settings)

    assert provider.get_model_calls == 1
    assert provider.cleanup_calls == 0


@pytest.mark.asyncio
async def test_prewarm_lock_serialises_concurrent_ensure_local_ready(monkeypatch):
    """Mirrors test_stt.py's test_load_lock_serialises_concurrent_get_model:
    two overlapping ensure_local_ready() calls for the same STTSettings must
    funnel through `_prewarm_lock` — the real load work happens exactly once."""
    call_count = {"n": 0}

    class _SlowFakeProvider:
        def __init__(self):
            self.is_loaded = False

        def _get_model(self):
            call_count["n"] += 1
            time.sleep(0.05)
            self.is_loaded = True

        def cleanup(self):
            pass

    provider = _SlowFakeProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    await asyncio.gather(
        local_setup.ensure_local_ready(settings),
        local_setup.ensure_local_ready(settings),
    )

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_ensure_local_ready_cleans_up_orphan_when_cache_cleared_without_a_mode_change(
    monkeypatch,
):
    """Reproduces the reviewer's exact RED-1 repro (Review — iteration 1,
    spec 015): an unrelated `PUT /settings` edit (e.g. an `initial_prompt`
    change) calls `clear_cache()` directly — bypassing `_prewarm_lock`
    entirely, via `sync_to_runtime()`'s `changed_stt` branch — while a
    prewarm load for provider A is still in flight, with `stt_settings.mode`
    staying LOCAL the entire time (never flipped). A mode-only recheck would
    see LOCAL throughout and skip cleanup, permanently leaking provider A's
    loaded model. The identity check must still catch it, and the second
    `ensure_local_ready()` call (blocked on `_prewarm_lock` the whole time)
    then creates and loads a second, distinct provider B into the now-empty
    cache — a redundant reload, but not a leak (see plan Risks)."""
    settings = STTSettings(mode=ProviderMode.LOCAL)
    cache: dict[str, _FakePrewarmProvider | None] = {"current": None}
    created: list[_FakePrewarmProvider] = []

    def _clear_cache_mid_load(self_provider):
        time.sleep(0.05)
        cache["current"] = None
        self_provider.is_loaded = True

    def _fake_get_provider(mode, s):
        if cache["current"] is None:
            impl = _clear_cache_mid_load if not created else None
            provider = _FakePrewarmProvider(get_model=impl)
            created.append(provider)
            cache["current"] = provider
        return cache["current"]

    monkeypatch.setattr("app.stt.get_provider", _fake_get_provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: cache["current"])
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    await asyncio.gather(
        local_setup.ensure_local_ready(settings),
        local_setup.ensure_local_ready(settings),
    )

    assert settings.mode == ProviderMode.LOCAL
    provider_a, provider_b = created
    assert provider_a is not provider_b
    assert provider_a.get_model_calls == 1
    assert provider_a.cleanup_calls == 1
    assert provider_b.get_model_calls == 1
    assert provider_b.cleanup_calls == 0
    assert cache["current"] is provider_b




@pytest.fixture
def isolated_crash_guard_root(tmp_path, monkeypatch):
    """Points `_crash_guard_path()` at a per-test `tmp_path` instead of the
    real `~/.justsay-dev`, mirroring test_app_paths.py's existing isolation
    pattern. Patched on the source (`app.core.app_paths.resolve_app_data_root`)
    since `_crash_guard_path()` imports it lazily on every call."""
    monkeypatch.setattr("app.core.app_paths.resolve_app_data_root", lambda: tmp_path)
    return tmp_path


def test_should_skip_prewarm_pure_function():
    """Pure decision function -- no filesystem access or mocks involved."""
    assert local_setup.should_skip_prewarm(0) is False
    assert local_setup.should_skip_prewarm(1) is False
    assert local_setup.should_skip_prewarm(2) is True
    assert local_setup.should_skip_prewarm(3) is True


def test_read_consecutive_incomplete_prewarms_fails_open_on_corrupt_file(
    isolated_crash_guard_root,
):
    guard_path = isolated_crash_guard_root / local_setup._CRASH_GUARD_FILENAME
    guard_path.write_bytes(b"not json {{{")

    assert local_setup._read_consecutive_incomplete_prewarms() == 0


@pytest.mark.prewarm
def test_maybe_prewarm_local_at_startup_noop_for_cloud_mode(
    isolated_crash_guard_root, monkeypatch
):
    """No-ops immediately for CLOUD mode without touching the filesystem at
    all -- no guard file is created under the monkeypatched, empty tmp_path
    data root.

    Marked `@pytest.mark.prewarm`: without it, backend/tests/conftest.py's
    autouse fixture would replace `maybe_prewarm_local_at_startup` itself
    with a no-op, making this test pass vacuously instead of exercising the
    real function (same reasoning as the existing
    `test_maybe_prewarm_local_is_noop_for_cloud_mode` above).
    """
    called = {"n": 0}

    async def _spy(stt_settings):
        called["n"] += 1

    monkeypatch.setattr(local_setup, "ensure_local_ready", _spy)
    settings = STTSettings(mode=ProviderMode.CLOUD)
    local_setup.maybe_prewarm_local_at_startup(settings)

    assert called["n"] == 0
    assert list(isolated_crash_guard_root.iterdir()) == []


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_maybe_prewarm_local_at_startup_marks_dirty_then_clears_on_completion(
    isolated_crash_guard_root, monkeypatch
):
    """The on-disk counter reads 1 immediately after
    `maybe_prewarm_local_at_startup()` returns (written synchronously, before
    the background task is scheduled) and 0 again once the wrapped
    `ensure_local_ready()` call completes, by any path."""
    gate = asyncio.Event()

    async def _blocking_ensure_local_ready(stt_settings):
        await gate.wait()

    monkeypatch.setattr(local_setup, "ensure_local_ready", _blocking_ensure_local_ready)
    settings = STTSettings(mode=ProviderMode.LOCAL)

    local_setup.maybe_prewarm_local_at_startup(settings)

    assert local_setup._read_consecutive_incomplete_prewarms() == 1

    gate.set()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending)

    assert local_setup._read_consecutive_incomplete_prewarms() == 0


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_maybe_prewarm_local_at_startup_routes_through_spawn_background_task(
    isolated_crash_guard_root, monkeypatch, spawn_spy
):
    """AC 8 (Spec 032): maybe_prewarm_local_at_startup() must schedule its
    _prewarm_then_clear_crash_guard() call through
    app.core.tasks.spawn_background_task() -- exactly once for LOCAL mode,
    and not at all for CLOUD mode.

    Uses the shared `spawn_spy` fixture (conftest.py) and awaits
    `spawn_spy.tasks` -- the exact task(s) it spawned -- rather than every
    task pending on the loop (GitHub review finding 1)."""

    async def _fake_ensure_local_ready(stt_settings):
        return None

    monkeypatch.setattr(local_setup, "ensure_local_ready", _fake_ensure_local_ready)

    local_setup.maybe_prewarm_local_at_startup(STTSettings(mode=ProviderMode.CLOUD))
    assert spawn_spy.names == []

    local_setup.maybe_prewarm_local_at_startup(STTSettings(mode=ProviderMode.LOCAL))
    await asyncio.gather(*spawn_spy.tasks)

    assert spawn_spy.names == ["local-stt-prewarm-startup"]


@pytest.mark.prewarm
def test_maybe_prewarm_local_at_startup_skips_after_max_consecutive_crashes(
    isolated_crash_guard_root, monkeypatch, caplog
):
    """Once the on-disk counter reaches MAX_CONSECUTIVE_INCOMPLETE_PREWARMS,
    ensure_local_ready() is never scheduled, the counter is left unchanged,
    and one log.warning(...) line names the count and the fallback/retry
    path."""
    local_setup._write_consecutive_incomplete_prewarms(
        local_setup.MAX_CONSECUTIVE_INCOMPLETE_PREWARMS
    )

    def _boom(stt_settings):
        raise AssertionError(
            "ensure_local_ready must not be scheduled once the crash-loop guard trips"
        )

    monkeypatch.setattr(local_setup, "ensure_local_ready", _boom)
    settings = STTSettings(mode=ProviderMode.LOCAL)

    with caplog.at_level(logging.WARNING, logger="app.stt.local_setup"):
        local_setup.maybe_prewarm_local_at_startup(settings)

    assert (
        local_setup._read_consecutive_incomplete_prewarms()
        == local_setup.MAX_CONSECUTIVE_INCOMPLETE_PREWARMS
    )
    assert "prewarm skipped at startup" in caplog.text.lower()
    assert str(local_setup.MAX_CONSECUTIVE_INCOMPLETE_PREWARMS) in caplog.text


@pytest.mark.prewarm
@pytest.mark.asyncio
async def test_maybe_prewarm_local_resets_crash_guard_counter_on_explicit_trigger(
    isolated_crash_guard_root, monkeypatch
):
    """A deliberate, user-initiated prewarm trigger (mode switch, settings
    edit, manual retry endpoint -- all routed through maybe_prewarm_local())
    resets the on-disk counter synchronously, before the dispatched task even
    has a chance to run -- decoupled from the automatic startup streak
    maybe_prewarm_local_at_startup() tracks."""
    local_setup._write_consecutive_incomplete_prewarms(1)

    async def _spy(stt_settings):
        pass

    monkeypatch.setattr(local_setup, "ensure_local_ready", _spy)
    settings = STTSettings(mode=ProviderMode.LOCAL)

    local_setup.maybe_prewarm_local(settings)

    assert local_setup._read_consecutive_incomplete_prewarms() == 0

    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending)




@pytest.mark.asyncio
async def test_await_local_ready_fast_path_when_already_loaded(monkeypatch):
    provider = _FakePrewarmProvider()
    provider.is_loaded = True
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)

    def _boom():
        raise AssertionError("_check_package_installed must not be called")

    monkeypatch.setattr(local_setup, "_check_package_installed", _boom)
    settings = STTSettings(mode=ProviderMode.LOCAL)

    result = await local_setup.await_local_ready(settings, timeout=5.0)

    assert result is True
    assert provider.get_model_calls == 0


@pytest.mark.asyncio
async def test_await_local_ready_returns_true_after_successful_load(monkeypatch):
    provider = _FakePrewarmProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    result = await local_setup.await_local_ready(settings, timeout=5.0)

    assert result is True
    assert provider.get_model_calls == 1


@pytest.mark.asyncio
async def test_await_local_ready_shares_prewarm_lock_no_second_get_model(monkeypatch):
    """AC 11: two overlapping await_local_ready() calls for the same
    not-yet-loaded provider must funnel through the shared _prewarm_lock --
    the real _get_model() work happens exactly once."""
    call_count = {"n": 0}

    class _SlowFakeProvider:
        def __init__(self):
            self.is_loaded = False

        def _get_model(self):
            call_count["n"] += 1
            time.sleep(0.05)
            self.is_loaded = True

        def cleanup(self):
            pass

    provider = _SlowFakeProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)
    results = await asyncio.gather(
        local_setup.await_local_ready(settings, timeout=5.0),
        local_setup.await_local_ready(settings, timeout=5.0),
    )

    assert call_count["n"] == 1
    assert results == [True, True]


@pytest.mark.asyncio
async def test_await_local_ready_raises_typed_timeout_on_stuck_load(monkeypatch):
    """AC 13: a genuinely stuck load must surface a clear, typed timeout
    rather than hanging the request path indefinitely."""

    class _StuckProvider:
        def __init__(self):
            self.is_loaded = False

        def _get_model(self):
            time.sleep(0.3)
            self.is_loaded = True

        def cleanup(self):
            pass

    provider = _StuckProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)

    with pytest.raises(local_setup.LocalReadinessTimeoutError):
        await local_setup.await_local_ready(settings, timeout=0.05)


@pytest.mark.asyncio
async def test_await_local_ready_returns_false_not_raises_when_mode_changes_mid_wait(
    monkeypatch,
):
    """Plan Risks / orchestrator correctness requirement: ensure_local_ready()'s
    own `mode != LOCAL` early-return guard racing in must surface as a plain
    `False`, never a LocalReadinessTimeoutError -- a request must never be worse
    off than before this barrier existed (it would have succeeded via the
    provider's own lazy _get_model() fallback)."""
    provider = _FakePrewarmProvider()
    cache = {"current": provider}
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: cache["current"])
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: cache["current"])
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: False)

    settings = STTSettings(mode=ProviderMode.LOCAL)

    def _install_and_flip_mode():
        settings.mode = ProviderMode.CLOUD
        cache["current"] = None
        return (0, "ok")

    monkeypatch.setattr(local_setup, "_run_pip_install", _install_and_flip_mode)

    result = await local_setup.await_local_ready(settings, timeout=5.0)

    assert result is False
    assert provider.get_model_calls == 0


@pytest.mark.asyncio
async def test_await_local_ready_returns_false_not_raises_when_cache_moves_on_mid_load(
    monkeypatch,
):
    """Same "must not raise" guarantee for the cache-identity-mismatch early
    return (an unrelated clear_cache() mid-load, mode untouched -- spec 015
    RED-1's exact scenario)."""
    settings = STTSettings(mode=ProviderMode.LOCAL)
    cache: dict[str, _FakePrewarmProvider | None] = {"current": None}

    def _flip_cache_during_load(self_provider):
        cache["current"] = None
        self_provider.is_loaded = True

    provider = _FakePrewarmProvider(get_model=_flip_cache_during_load)
    cache["current"] = provider
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: cache["current"])
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: cache["current"])
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    result = await local_setup.await_local_ready(settings, timeout=5.0)

    assert result is False
    assert provider.get_model_calls == 1
    assert provider.cleanup_calls == 1




@pytest.mark.asyncio
async def test_timeout_then_retry_joins_in_flight_load_instead_of_starting_a_second_one(
    monkeypatch,
):
    """Reproduces the exact bug: asyncio.wait_for's timeout cancels the
    WAITING coroutine, not the worker thread _get_model() actually runs on
    (a thread cannot be cancelled). Without tracking the in-flight load
    separately from _prewarm_lock, the lock frees up while the first
    _get_model() call keeps running, and a retry that arrives while it is
    still in flight would start a genuinely SECOND _get_model() call --
    "Two heavy loads of the same provider running concurrently."

    Uses a real threading.Event (not a mock) so the fake _get_model()
    genuinely blocks its OWN worker thread, exactly like the real
    WhisperModel(...)/whisper-server load this simulates -- a mocked sleep
    would return before wait_for's timeout could ever race it.
    """
    call_count = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    class _SlowProvider:
        def __init__(self):
            self.is_loaded = False

        def _get_model(self):
            call_count["n"] += 1
            started.set()
            release.wait(timeout=5)
            self.is_loaded = True

        def cleanup(self):
            pass

    provider = _SlowProvider()
    monkeypatch.setattr("app.stt.get_provider", lambda mode, s: provider)
    monkeypatch.setattr("app.stt.peek_local_provider", lambda: provider)
    monkeypatch.setattr(local_setup, "_check_package_installed", lambda: True)

    settings = STTSettings(mode=ProviderMode.LOCAL)

    with pytest.raises(local_setup.LocalReadinessTimeoutError):
        await local_setup.await_local_ready(settings, timeout=0.05)

    assert await asyncio.to_thread(started.wait, 2), "the worker thread never started"
    assert call_count["n"] == 1

    async def _release_soon():
        await asyncio.sleep(0.1)
        release.set()

    asyncio.create_task(_release_soon())
    result = await local_setup.await_local_ready(settings, timeout=5.0)

    assert call_count["n"] == 1, (
        "a second _get_model() call was started while the first was still in flight"
    )
    assert result is True
