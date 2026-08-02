"""Pure command-construction and path-resolution helpers for the
whisper.cpp ``whisper-server`` binary -- Vulkan-backed on Windows,
Metal-backed on macOS Apple Silicon. The GPU backend is a property of the
compiled binary, so the only platform-dependent thing here is which vendor
directory and file name to look for.

Kept separate from ``local_whisper_cpp.py`` so ``build_server_argv()`` is
unit-testable with zero process/network I/O -- mirrors the *intent* of the
sibling project ``local-llm-helper``'s ADR-013 ``buildCmd()`` pattern: a
pure function that builds the argument list, kept apart from the actual
spawn call. Python's ``subprocess.Popen(argv, shell=False)`` never goes
through a shell and needs no manual quoting -- the isolation here buys
testability of the argument *set* (right flags/paths in the right order),
not a quoting-bug fix that doesn't exist in this invocation path. See
``docs/adr/011-whisper-cpp-vulkan-stt-provider.md``.
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
    """The per-platform vendor directory leaf, or ``None`` on a platform
    that ships no whisper.cpp binary.

    Public because ``backend/tests/test_build_definitions.py`` compares this
    mapping against the resource directories the Tauri platform configs
    declare, so the two sides cannot drift apart silently.
    """
    return VENDOR_DIR_NAMES.get(sys.platform)


def _binary_name() -> str:
    return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"


BUILD_SCRIPT_NAMES: dict[str, str] = {
    "win32": "backend/scripts/build_whisper_cpp_vulkan.ps1",
    "darwin": "backend/scripts/build_whisper_cpp_metal.sh",
}


def binary_not_found_message() -> str:
    """The one wording for "no whisper-server here", shared by the provider's
    load failure and ``local_setup.ensure_local_ready()``'s prewarm error so
    the two cannot name different build scripts on the same platform.
    """
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
    """Resolution order: ``JUSTSAY_WHISPER_CPP_BIN`` env override -> the
    bundled resource directory sibling to ``sys.executable`` when frozen ->
    a local dev-vendor directory (``backend/vendor/<vendor_dir_name()>/``)
    -> ``None``.

    Each source is accepted only when the resolved file actually exists --
    an override/frozen path that doesn't resolve degrades to the next
    source rather than failing outright, mirroring gpu_probe.py's
    degrade-only chain philosophy. A platform with no entry in
    ``VENDOR_DIR_NAMES`` degrades to ``None`` after the env override rather
    than constructing a directory name that means nothing there.
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

    if getattr(sys, "frozen", False):
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
