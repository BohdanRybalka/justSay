"""Pure command-construction and path-resolution helpers for the
whisper.cpp Vulkan-backed ``whisper-server`` binary.

Kept separate from ``local_vulkan.py`` so ``build_server_argv()`` is
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

_ENV_VAR = "JUSTSAY_WHISPER_CPP_BIN"

_BINARY_NAME = "whisper-server.exe" if os.name == "nt" else "whisper-server"

_DEV_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "whisper-cpp-vulkan"


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
    a local dev-vendor directory (``backend/vendor/whisper-cpp-vulkan/``) ->
    ``None``.

    Each source is accepted only when the resolved file actually exists --
    an override/frozen path that doesn't resolve degrades to the next
    source rather than failing outright, mirroring gpu_probe.py's
    degrade-only chain philosophy.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate

    if getattr(sys, "frozen", False):
        resource_dir = Path(sys.executable).resolve().parent.parent
        candidate = resource_dir / "whisper-cpp-vulkan" / _BINARY_NAME
        if candidate.is_file():
            return candidate

    candidate = _DEV_VENDOR_DIR / _BINARY_NAME
    if candidate.is_file():
        return candidate

    return None


def resolve_model_path(model_size: str) -> Path:
    """Where the lazy-downloaded GGML model for ``model_size`` lives (or
    will live once downloaded). Pure path arithmetic -- no existence check,
    no I/O.
    """
    return Path.home() / ".justsay" / "models" / "whisper-cpp" / f"ggml-{model_size}.bin"
