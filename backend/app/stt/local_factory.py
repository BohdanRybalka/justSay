"""Platform-aware local STT provider selection.

Local mode runs on `WhisperCppServerSTTProvider` (macOS Apple Silicon via
Metal, Windows AMD/Intel via Vulkan) or `LocalSTTProvider` (faster-whisper,
CUDA on NVIDIA, CPU otherwise). `get_local_provider_kind` owns that decision
and `get_local_provider_class` dispatches on it. No third-party import at
module level, so this module imports on any platform whatever is installed.
"""

import os
from enum import Enum
from typing import TYPE_CHECKING

from app.stt.base import STTProvider

if TYPE_CHECKING:
    from app.core.gpu_probe import GpuVendor


class LocalProviderKind(str, Enum):
    FASTER_WHISPER = "faster_whisper"
    WHISPER_CPP_SERVER = "whisper_cpp_server"


_ACCELERATED_DEVICES: dict[LocalProviderKind, frozenset[str]] = {
    LocalProviderKind.FASTER_WHISPER: frozenset({"cuda"}),
    LocalProviderKind.WHISPER_CPP_SERVER: frozenset({"metal", "vulkan"}),
}


def is_accelerated_device(device: str, kind: LocalProviderKind) -> bool:
    """Whether the provider that will load actually reaches a GPU on `device`.

    faster-whisper accelerates only on `"cuda"`, whisper.cpp only on
    `"metal"`/`"vulkan"`, and an unlisted kind accelerates nothing.
    """
    return device in _ACCELERATED_DEVICES.get(kind, frozenset())


def compute_type_for_device(device: str, kind: LocalProviderKind) -> str:
    """The compute type a device implies for the provider that will load it.

    `"float16"` on an accelerated device, `"int8"` on every other one,
    including an unrecognized `whisper_device`.
    """
    return "float16" if is_accelerated_device(device, kind) else "int8"


def is_macos_arm64() -> bool:
    """True only when running natively on Apple Silicon.

    Rosetta-x86 Python reports `"x86_64"` and is therefore False, as is
    macOS Intel.
    """
    import sys

    if sys.platform != "darwin":
        return False
    import platform

    return platform.machine() == "arm64"


def get_local_provider_kind(vendor: "GpuVendor | None" = None) -> LocalProviderKind:
    """Resolve which local STT provider kind applies to this machine.

    macOS arm64 and Windows-with-AMD/Intel get `WHISPER_CPP_SERVER`, anything
    else `FASTER_WHISPER`. Pass `vendor` to reuse an existing `probe_gpu()`.
    """
    if is_macos_arm64():
        return LocalProviderKind.WHISPER_CPP_SERVER

    if os.name == "nt":
        from app.core.gpu_probe import GpuVendor

        if vendor is None:
            from app.core.gpu_probe import probe_gpu

            vendor = probe_gpu().vendor
        if vendor in (GpuVendor.AMD, GpuVendor.INTEL):
            return LocalProviderKind.WHISPER_CPP_SERVER

    return LocalProviderKind.FASTER_WHISPER


LOCAL_STATUS_CONTRACT: tuple[str, ...] = ("_get_model", "is_loaded", "last_load_error")


def get_local_provider_class() -> type[STTProvider]:
    """The concrete provider class Local mode runs on this machine, contract checked.

    Raises `TypeError` unless that class declares every member of
    `LOCAL_STATUS_CONTRACT`, which no base class supplies (ADR 075).
    """
    if get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER:
        from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider

        provider_class: type[STTProvider] = WhisperCppServerSTTProvider
    else:
        from app.stt.local import LocalSTTProvider

        provider_class = LocalSTTProvider

    missing = [name for name in LOCAL_STATUS_CONTRACT if not hasattr(provider_class, name)]
    if missing:
        raise TypeError(
            f"{provider_class.__name__} is the local STT provider on this machine "
            f"but does not declare {missing}, which GET /stt/local/status and "
            f"POST /stt/local/load read"
        )
    return provider_class
