"""Pure command-construction and path-resolution helpers for the whisper.cpp
``whisper-server`` binary -- Vulkan on Windows, Metal on macOS Apple Silicon.

The GPU backend is a property of the compiled binary, so the only
platform-dependent thing here is which vendor directory and file name to look
for. Kept apart from ``local_whisper_cpp.py`` so ``build_server_argv()`` is
unit-testable with zero process or network I/O (ADR 011).
"""

import os
import sys
from pathlib import Path

_WHISPER_CPP_BIN_ENV_VAR = "JUSTSAY_WHISPER_CPP_BIN"

VENDOR_DIR_NAMES: dict[str, str] = {
    "win32": "whisper-cpp-vulkan",
    "darwin": "whisper-cpp-metal",
}

_VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor"


def vendor_dir_name() -> str | None:
    """The per-platform vendor directory leaf, or ``None`` on a platform that
    ships no whisper.cpp binary. Public so the build-definition test can
    compare it against the resource directories the Tauri configs declare.
    """
    return VENDOR_DIR_NAMES.get(sys.platform)


def _binary_name() -> str:
    return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"


BUILD_SCRIPT_NAMES: dict[str, str] = {
    "win32": "backend/scripts/build_whisper_cpp_vulkan.ps1",
    "darwin": "backend/scripts/build_whisper_cpp_metal.sh",
}


_INSTALLED_BUILD_BINARY_MISSING = (
    "The local speech engine is missing from this installation of JustSay. "
    "Reinstall the app, or use Cloud mode until it is back."
)

_BIN_OVERRIDE_POINTS_AT_NOTHING = (
    f"{_WHISPER_CPP_BIN_ENV_VAR} is set, but there is no file at that path. "
    "Point it at a whisper-server binary, or unset it to use the bundled one."
)


def _is_frozen_build() -> bool:
    """Whether this process is a PyInstaller bundle rather than a source checkout."""
    return bool(getattr(sys, "frozen", False))


def binary_not_found_message() -> str:
    """The one wording for "no whisper-server here", shared by every reader.

    Follows `resolve_binary_path`'s own resolution order, so the sentence names
    the source that failed: the env override, the installed bundle, the checkout.
    """
    if os.environ.get(_WHISPER_CPP_BIN_ENV_VAR):
        return _BIN_OVERRIDE_POINTS_AT_NOTHING

    if _is_frozen_build():
        return _INSTALLED_BUILD_BINARY_MISSING

    script = BUILD_SCRIPT_NAMES.get(sys.platform, BUILD_SCRIPT_NAMES["win32"])
    return (
        "whisper-server binary not found. Set JUSTSAY_WHISPER_CPP_BIN, "
        f"or run {script} for local dev."
    )


def build_server_argv(binary_path: Path, model_path: Path, host: str, port: int) -> list[str]:
    """Pure -- no I/O, no subprocess spawn. Returns a ``list[str]`` (never a
    shell string), so a binary/model path containing a space survives as a
    single, unmangled argv element.
    """
    return [
        str(binary_path),
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
    ]


def resolve_binary_path() -> Path | None:
    """Resolution order: ``JUSTSAY_WHISPER_CPP_BIN`` env override -> the bundled
    resource directory beside ``sys.executable`` when frozen -> the dev-vendor
    directory -> ``None``. A source is accepted only when the resolved file
    exists, so one that does not resolve degrades to the next rather than fails.
    """
    override = os.environ.get(_WHISPER_CPP_BIN_ENV_VAR)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate

    vendor_dir = vendor_dir_name()
    if vendor_dir is None:
        return None

    binary_name = _binary_name()

    if _is_frozen_build():
        resource_dir = Path(sys.executable).resolve().parent.parent
        candidate = resource_dir / vendor_dir / binary_name
        if candidate.is_file():
            return candidate

    candidate = _VENDOR_ROOT / vendor_dir / binary_name
    if candidate.is_file():
        return candidate

    return None


def resolve_model_path(model_size: str) -> Path:
    """Where the lazy-downloaded GGML model for ``model_size`` lives (or
    will live once downloaded). Pure path arithmetic -- no existence check,
    no I/O.
    """
    return Path.home() / ".justsay" / "models" / "whisper-cpp" / f"ggml-{model_size}.bin"
