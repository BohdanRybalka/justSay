from unittest.mock import MagicMock, patch

import pytest

from app.core.utils import sse_event
from app.stt.config import STTSettings
from app.stt import local_setup
from app.stt.local_setup import (
    LocalSttStatus,
    _check_package_installed,
    _detect_gpu,
    _local_extras,
    check_status,
    install_local_packages,
)


# --- check_status ---


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
    assert status.model_loaded is False  # nothing loaded in tests
    assert status.last_error is None     # no failure yet
    assert status.model_ram_mb is None   # not loaded → no RAM
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
    assert status.model_loaded is False  # short-circuited because package missing


def test_check_status_reports_amd_gpu_name_and_vendor_but_not_available():
    """AMD is detected (name + vendor populated) but gpu_available stays False
    — faster-whisper has no AMD backend (spec 014)."""
    settings = STTSettings(whisper_device="auto")

    with _apply(_patches(True, (False, "AMD Radeon RX 5700 XT", "amd"))):
        status = check_status(settings)

    assert status.gpu_available is False
    assert status.gpu_vendor == "amd"
    assert status.gpu_name == "AMD Radeon RX 5700 XT"
    assert status.device == "cpu"  # auto + not available -> cpu


def test_check_status_surfaces_last_load_error():
    """When _get_model latched an error, status.last_error must contain it."""
    from app.stt import clear_cache as clear_stt_cache
    from app.stt.local import LocalSTTProvider

    clear_stt_cache()
    settings = STTSettings()
    # Force-create provider, then set its instance-level error.
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
    from app.stt import clear_cache as clear_stt_cache, get_local_load_error

    clear_stt_cache()
    settings = STTSettings()
    assert get_local_load_error(settings) is None


# --- _check_package_installed ---


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


# --- _detect_gpu ---


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


# --- sse_event formatting ---


def test_sse_format_is_event_plus_data():
    out = sse_event("progress", {"status": "downloading"})
    assert out.startswith("event: progress\n")
    assert 'data: {"status": "downloading"}' in out
    assert out.endswith("\n\n")


# --- install_local_packages ---


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


# --- macOS arm64 platform path (Plan 019) ---


def test_local_extras_returns_local_on_non_mac():
    with patch("app.stt.local_setup.is_macos_arm64", return_value=False):
        assert _local_extras() == "local"


def test_local_extras_returns_local_mac_on_macos_arm64():
    with patch("app.stt.local_setup.is_macos_arm64", return_value=True):
        assert _local_extras() == "local-mac"


def test_run_pip_install_uses_local_mac_extras_on_macos_arm64(monkeypatch):
    """Pip command must include `.[local-mac]` (not `.[local]`) on M1+."""
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(local_setup.subprocess, "run", _fake_run)
    code, _ = local_setup._run_pip_install()
    assert code == 0
    assert ".[local-mac]" in captured["cmd"]


def test_check_package_installed_imports_mlx_whisper_on_macos_arm64(monkeypatch):
    fake_module = MagicMock()
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    with patch.dict("sys.modules", {"mlx_whisper": fake_module}):
        assert _check_package_installed() is True


def test_check_package_installed_false_when_mlx_whisper_missing_on_macos_arm64(monkeypatch):
    import builtins

    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _check_package_installed() is False


def test_detect_gpu_reports_apple_silicon_on_macos_arm64(monkeypatch):
    """On M1+ we must return the MLX/Metal label without importing torch.

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
    assert name == "Apple Silicon (MLX/Metal)"
    assert vendor == "apple"


def test_check_status_macos_arm64_reports_mlx_device_and_bfloat16(monkeypatch):
    """On macOS arm64 `check_status` must return device='mlx', compute_type='bfloat16'."""
    settings = STTSettings(whisper_model_size="large-v3-turbo")
    monkeypatch.setattr("app.stt.local_setup.is_macos_arm64", lambda: True)

    with _apply(_patches(True, (True, "Apple Silicon (MLX/Metal)", "apple"))):
        status = check_status(settings)

    assert status.device == "mlx"
    assert status.compute_type == "bfloat16"
    assert status.gpu_available is True
    assert status.gpu_name == "Apple Silicon (MLX/Metal)"
    assert status.gpu_vendor == "apple"
