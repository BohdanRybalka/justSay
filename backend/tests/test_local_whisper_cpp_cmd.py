"""Tests for `app.stt.local_whisper_cpp_cmd` -- pure argv construction and
binary/model path resolution. No subprocess spawn, no network I/O.

Every binary-resolution test pins `sys.platform` explicitly. Resolution is
platform-keyed (`VENDOR_DIR_NAMES`), so a test inheriting the host's platform
would pass on a Windows dev box and fail on `ubuntu-latest` CI, where
`vendor_dir_name()` is `None` and `resolve_binary_path()` short-circuits.
"""

import inspect
import sys
from pathlib import Path

import pytest

import app.stt.local_whisper_cpp as local_whisper_cpp_module
import app.stt.local_whisper_cpp_cmd as local_whisper_cpp_cmd_module
from app.core.app_paths import resolve_app_data_root
from app.stt.local_whisper_cpp_cmd import (
    VENDOR_DIR_NAMES,
    _binary_name,
    build_server_argv,
    resolve_binary_path,
    resolve_model_path,
    vendor_dir_name,
)


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


def test_no_shell_true_anywhere_in_provider_or_cmd_module():
    """Static check: neither module ever spawns via a shell string."""
    forbidden = "shell" + "=" + "True"
    for module in (local_whisper_cpp_module, local_whisper_cpp_cmd_module):
        source = inspect.getsource(module)
        assert forbidden not in source




PLATFORMS = [
    ("win32", "whisper-cpp-vulkan", "whisper-server.exe"),
    ("darwin", "whisper-cpp-metal", "whisper-server"),
]


def _pin_platform(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)


def test_vendor_dir_names_pin_both_shipped_platforms():
    """The Windows value is what the shipped Windows build already resolves,
    and what `WhisperCppServerSTTProvider.model_name` -- persisted into every
    history row -- is derived from. Changing it splits existing Windows
    history across two labels for one engine.
    """
    assert VENDOR_DIR_NAMES["win32"] == "whisper-cpp-vulkan"
    assert VENDOR_DIR_NAMES["darwin"] == "whisper-cpp-metal"


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_vendor_dir_name_and_binary_name_per_platform(monkeypatch, platform, vendor_dir, binary):
    _pin_platform(monkeypatch, platform)
    assert vendor_dir_name() == vendor_dir
    assert _binary_name() == binary


def test_vendor_dir_name_is_none_on_an_unsupported_platform(monkeypatch):
    _pin_platform(monkeypatch, "linux")
    assert vendor_dir_name() is None


def test_resolve_binary_path_returns_none_on_an_unsupported_platform(monkeypatch):
    """Linux ships no whisper.cpp binary at all: resolution must degrade to
    `None` rather than construct `vendor/None/whisper-server`."""
    _pin_platform(monkeypatch, "linux")
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert resolve_binary_path() is None


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_env_override_wins(
    tmp_path, monkeypatch, platform, vendor_dir, binary
):
    _pin_platform(monkeypatch, platform)
    override = tmp_path / binary
    override.write_bytes(b"")
    monkeypatch.setenv("JUSTSAY_WHISPER_CPP_BIN", str(override))

    assert resolve_binary_path() == override


def test_resolve_binary_path_env_override_wins_even_on_an_unsupported_platform(
    tmp_path, monkeypatch
):
    """The override is checked before `vendor_dir_name()`, so pointing it at a
    self-built binary still works where no vendor directory is defined."""
    _pin_platform(monkeypatch, "linux")
    override = tmp_path / "whisper-server"
    override.write_bytes(b"")
    monkeypatch.setenv("JUSTSAY_WHISPER_CPP_BIN", str(override))

    assert resolve_binary_path() == override


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_env_override_missing_file_falls_through(
    tmp_path, monkeypatch, platform, vendor_dir, binary
):
    """An override pointing at a nonexistent file degrades to the next
    source rather than being trusted blindly -- mirrors gpu_probe.py's
    degrade-only chain philosophy."""
    _pin_platform(monkeypatch, platform)
    monkeypatch.setenv("JUSTSAY_WHISPER_CPP_BIN", str(tmp_path / "does-not-exist"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(local_whisper_cpp_cmd_module, "_VENDOR_ROOT", tmp_path / "no-vendor-root")

    assert resolve_binary_path() is None


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_frozen_resource_dir(
    tmp_path, monkeypatch, platform, vendor_dir, binary
):
    """sys.executable == <resource_dir>/justsay-backend/justsay-backend[.exe];
    the vendor directory is bundled as a SIBLING of justsay-backend/, so the
    resolved directory is two levels up from sys.executable, not one."""
    _pin_platform(monkeypatch, platform)
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    resource_dir = tmp_path / "resources"
    sidecar_dir = resource_dir / "justsay-backend"
    sidecar_dir.mkdir(parents=True)
    fake_exe = sidecar_dir / "justsay-backend.exe"
    fake_exe.write_bytes(b"")

    bundled_dir = resource_dir / vendor_dir
    bundled_dir.mkdir()
    bundled = bundled_dir / binary
    bundled.write_bytes(b"")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert resolve_binary_path() == bundled


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_dev_vendor_dir(tmp_path, monkeypatch, platform, vendor_dir, binary):
    _pin_platform(monkeypatch, platform)
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)

    vendor_root = tmp_path / "vendor"
    dev_dir = vendor_root / vendor_dir
    dev_dir.mkdir(parents=True)
    dev_binary = dev_dir / binary
    dev_binary.write_bytes(b"")
    monkeypatch.setattr(local_whisper_cpp_cmd_module, "_VENDOR_ROOT", vendor_root)

    assert resolve_binary_path() == dev_binary


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_ignores_the_other_platforms_vendor_dir(
    tmp_path, monkeypatch, platform, vendor_dir, binary
):
    """A Metal binary must never satisfy a Windows resolution, or vice versa
    -- the two directories are not interchangeable."""
    _pin_platform(monkeypatch, platform)
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)

    other_dir = next(d for _, d, _ in PLATFORMS if d != vendor_dir)
    vendor_root = tmp_path / "vendor"
    (vendor_root / other_dir).mkdir(parents=True)
    (vendor_root / other_dir / binary).write_bytes(b"")
    monkeypatch.setattr(local_whisper_cpp_cmd_module, "_VENDOR_ROOT", vendor_root)

    assert resolve_binary_path() is None


@pytest.mark.parametrize("platform,vendor_dir,binary", PLATFORMS)
def test_resolve_binary_path_returns_none_when_nothing_resolves(
    tmp_path, monkeypatch, platform, vendor_dir, binary
):
    _pin_platform(monkeypatch, platform)
    monkeypatch.delenv("JUSTSAY_WHISPER_CPP_BIN", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(local_whisper_cpp_cmd_module, "_VENDOR_ROOT", tmp_path / "nope")

    assert resolve_binary_path() is None




def test_resolve_model_path_is_pure_path_arithmetic():
    path = resolve_model_path("large-v3-turbo")
    assert path == Path.home() / ".justsay" / "models" / "whisper-cpp" / "ggml-large-v3-turbo.bin"


def test_model_cache_stays_shared_between_dev_and_production():
    """The one app-data consumer that deliberately ignores the dev/prod split.

    ADR 012 settled this: downloaded GGML weights are multi-GB, identical
    across roots, and carry no personal data, so dev runs share the production
    cache instead of re-downloading into `~/.justsay-dev`. ADR 014's consumer
    inventory tags it `exception` for the same reason.

    This asserts the decision rather than the arithmetic the sibling test
    covers. The session already points `resolve_app_data_root()` at a tmp
    directory, so "the model path is not under the app-data root" is exactly
    the property that distinguishes the exception from every other consumer --
    and it holds without naming any real directory. Routing
    `resolve_model_path` through `resolve_app_data_root()` reads like a
    tidy-up and would silently orphan every already-downloaded model, so it
    fails here naming the ADR instead of as an unexplained path mismatch.
    """
    app_data_root = resolve_app_data_root()

    path = resolve_model_path("tiny")

    assert not path.is_relative_to(app_data_root), (
        f"resolve_model_path now follows the app-data root ({app_data_root}). "
        "That bypass is a deliberate ADR 012 exception, not an oversight -- "
        "changing it relocates multi-GB GGML models and orphans existing "
        "downloads. Amend ADR 012, ADR 014's consumer inventory and "
        "test_data_isolation's inventory tag together, or revert."
    )


def test_resolve_model_path_does_not_touch_filesystem():
    path = resolve_model_path("tiny")
    assert path.name == "ggml-tiny.bin"
