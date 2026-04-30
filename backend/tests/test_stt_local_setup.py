from unittest.mock import MagicMock, patch

import pytest

from app.stt.config import STTSettings
from app.stt import local_setup
from app.stt.local_setup import (
    LocalSttStatus,
    _check_package_installed,
    _detect_gpu,
    _sse,
    check_status,
    install_local_packages,
)


# --- check_status ---


def test_check_status_reports_installed_package():
    settings = STTSettings(whisper_model_size="large-v3-turbo", whisper_device="auto")

    with patch.object(local_setup, "_check_package_installed", return_value=True), patch.object(
        local_setup, "_detect_gpu", return_value=(False, None)
    ):
        status = check_status(settings)

    assert isinstance(status, LocalSttStatus)
    assert status.package_installed is True
    assert status.model_name == "large-v3-turbo"
    assert status.gpu_available is False
    assert status.gpu_name is None
    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_uses_cuda_when_gpu_auto():
    settings = STTSettings(whisper_device="auto")

    with patch.object(local_setup, "_check_package_installed", return_value=True), patch.object(
        local_setup, "_detect_gpu", return_value=(True, "NVIDIA GeForce RTX 3060")
    ):
        status = check_status(settings)

    assert status.gpu_available is True
    assert status.gpu_name == "NVIDIA GeForce RTX 3060"
    assert status.device == "cuda"
    assert status.compute_type == "float16"


def test_check_status_respects_explicit_cpu_device():
    settings = STTSettings(whisper_device="cpu")

    with patch.object(local_setup, "_check_package_installed", return_value=True), patch.object(
        local_setup, "_detect_gpu", return_value=(True, "RTX 3060")
    ):
        status = check_status(settings)

    assert status.device == "cpu"
    assert status.compute_type == "int8"


def test_check_status_reports_missing_package():
    settings = STTSettings()

    with patch.object(local_setup, "_check_package_installed", return_value=False), patch.object(
        local_setup, "_detect_gpu", return_value=(False, None)
    ):
        status = check_status(settings)

    assert status.package_installed is False


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


# --- _sse formatting ---


def test_sse_format_is_event_plus_data():
    out = _sse("progress", {"status": "downloading"})
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
