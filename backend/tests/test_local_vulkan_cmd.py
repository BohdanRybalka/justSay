"""Tests for `app.stt.local_vulkan_cmd` -- pure argv construction and
binary/model path resolution. No subprocess spawn, no network I/O.
"""

import inspect
import os
import sys
from pathlib import Path

import app.stt.local_vulkan as local_vulkan_module
import app.stt.local_vulkan_cmd as local_vulkan_cmd_module
from app.stt.local_vulkan_cmd import build_server_argv, resolve_binary_path, resolve_model_path

# --- build_server_argv ---


def test_build_server_argv_returns_list_of_str():
    argv = build_server_argv(
        binary_path=Path("whisper-server.exe"),
        model_path=Path("ggml-large-v3-turbo.bin"),
        host="127.0.0.1",
        port=8878,
    )
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)


def test_build_server_argv_contains_expected_flags():
    argv = build_server_argv(
        binary_path=Path("whisper-server.exe"),
        model_path=Path("ggml-large-v3-turbo.bin"),
        host="127.0.0.1",
        port=8878,
    )
    assert argv[0] == "whisper-server.exe"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "ggml-large-v3-turbo.bin"
    assert "--host" in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv
    assert argv[argv.index("--port") + 1] == "8878"


def test_build_server_argv_preserves_space_containing_paths_as_single_element():
    """A path containing a space must survive as ONE unmangled list element
    -- proof that no manual quoting/concatenation happens inside this pure
    function (Python's `subprocess.Popen(argv, shell=False)` needs none).
    """
    binary = Path("C:\\Program Files\\JustSay\\whisper-cpp-vulkan\\whisper-server.exe")
    model = Path("C:\\Users\\Some User\\.justsay\\models\\whisper-cpp\\ggml-large-v3-turbo.bin")
    argv = build_server_argv(binary_path=binary, model_path=model, host="127.0.0.1", port=8878)

    assert str(binary) in argv
    assert str(model) in argv
    # No element got split/mangled on the internal space.
    assert all(" " not in part or part in (str(binary), str(model)) for part in argv)


def test_build_server_argv_is_pure_no_io():
    """Calling it must not touch the filesystem or spawn anything -- paths
    that don't exist on disk are accepted without error."""
    argv = build_server_argv(
        binary_path=Path("/does/not/exist/whisper-server"),
        model_path=Path("/does/not/exist/ggml-large-v3-turbo.bin"),
        host="127.0.0.1",
        port=8878,
    )
    assert isinstance(argv, list)


def test_no_shell_true_anywhere_in_local_vulkan_or_cmd_module():
    """Static check: neither module ever spawns via a shell string."""
    forbidden = "shell" + "=" + "True"
    for module in (local_vulkan_module, local_vulkan_cmd_module):
        source = inspect.getsource(module)
        assert forbidden not in source


# --- resolve_binary_path ---


def _binary_name() -> str:
    return "whisper-server.exe" if os.name == "nt" else "whisper-server"


def test_resolve_binary_path_env_override_wins(tmp_path, monkeypatch):
    binary = tmp_path / _binary_name()
    binary.write_bytes(b"")
    monkeypatch.setenv("JUSTSAY_WHISPER_CPP_BIN", str(binary))

    assert resolve_binary_path() == binary


def test_resolve_binary_path_env_override_missing_file_falls_through(tmp_path, monkeypatch):
    """An override pointing at a nonexistent file degrades to the next
    source rather than being trusted blindly -- mirrors gpu_probe.py's
    degrade-only chain philosophy."""
    monkeypatch.setenv("JUSTSAY_WHISPER_CPP_BIN", str(tmp_path / "does-not-exist.exe"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(local_vulkan_cmd_module, "_DEV_VENDOR_DIR", tmp_path / "no-dev-vendor-dir")

    assert resolve_binary_path() is None


def test_resolve_binary_path_frozen_resource_dir(tmp_path, monkeypatch):
    """sys.executable == <resource_dir>/justsay-backend/justsay-backend.exe;
    whisper-cpp-vulkan/ is bundled as a SIBLING of justsay-backend/, so the
    resolved directory is two levels up from sys.executable, not one."""
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    resource_dir = tmp_path / "resources"
    sidecar_dir = resource_dir / "justsay-backend"
    sidecar_dir.mkdir(parents=True)
    fake_exe = sidecar_dir / "justsay-backend.exe"
    fake_exe.write_bytes(b"")

    vulkan_dir = resource_dir / "whisper-cpp-vulkan"
    vulkan_dir.mkdir()
    binary = vulkan_dir / _binary_name()
    binary.write_bytes(b"")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert resolve_binary_path() == binary


def test_resolve_binary_path_dev_vendor_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)

    dev_dir = tmp_path / "vendor" / "whisper-cpp-vulkan"
    dev_dir.mkdir(parents=True)
    binary = dev_dir / _binary_name()
    binary.write_bytes(b"")
    monkeypatch.setattr(local_vulkan_cmd_module, "_DEV_VENDOR_DIR", dev_dir)

    assert resolve_binary_path() == binary


def test_resolve_binary_path_returns_none_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(local_vulkan_cmd_module, "_DEV_VENDOR_DIR", tmp_path / "nope")

    assert resolve_binary_path() is None


# --- resolve_model_path ---


def test_resolve_model_path_is_pure_path_arithmetic():
    path = resolve_model_path("large-v3-turbo")
    assert path == Path.home() / ".justsay" / "models" / "whisper-cpp" / "ggml-large-v3-turbo.bin"


def test_resolve_model_path_does_not_touch_filesystem():
    # Must not raise even though nothing on disk exists at this path.
    path = resolve_model_path("tiny")
    assert path.name == "ggml-tiny.bin"
