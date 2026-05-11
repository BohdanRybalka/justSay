from unittest.mock import MagicMock, patch

import pytest

from app.core.utils import sse_event
from app.stt.config import STTSettings
from app.stt import local_setup
from app.stt.local_setup import (
    LocalSttStatus,
    _check_package_installed,
    _detect_gpu,
    check_status,
    install_local_packages,
)


# --- check_status ---


def _patches(installed: bool, gpu: tuple[bool, str | None]):
    """Standard patch set: stub package detection + GPU detection."""
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

    with _apply(_patches(True, (False, None))):
        status = check_status(settings)

    assert isinstance(status, LocalSttStatus)
    assert status.package_installed is True
    assert status.model_loaded is False  # nothing loaded in tests
    assert status.last_error is None     # no failure yet
    assert status.model_ram_mb is None   # not loaded → no RAM
    assert status.model_name == "large-v3-turbo"
    assert status.gpu_available is False
    assert status.gpu_name is None
    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_uses_cuda_when_gpu_auto():
    settings = STTSettings(whisper_device="auto")

    with _apply(_patches(True, (True, "NVIDIA GeForce RTX 3060"))):
        status = check_status(settings)

    assert status.gpu_available is True
    assert status.gpu_name == "NVIDIA GeForce RTX 3060"
    assert status.device == "cuda"
    assert status.compute_type == "float16"


def test_check_status_respects_explicit_cpu_device():
    settings = STTSettings(whisper_device="cpu")

    with _apply(_patches(True, (True, "RTX 3060"))):
        status = check_status(settings)

    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_reports_missing_package():
    settings = STTSettings()

    with _apply(_patches(False, (False, None))):
        status = check_status(settings)

    assert status.package_installed is False
    assert status.model_loaded is False  # short-circuited because package missing


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
        with _apply(_patches(True, (False, None))):
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


def test_detect_gpu_returns_false_when_torch_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    available, name = _detect_gpu()
    assert available is False
    assert name is None


def test_detect_gpu_returns_name_when_cuda_available():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "RTX 4090"

    with patch.dict("sys.modules", {"torch": fake_torch}):
        available, name = _detect_gpu()

    assert available is True
    assert name == "RTX 4090"


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
