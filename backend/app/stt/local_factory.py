"""Platform-aware local STT provider selection.

Three concrete `STTProvider`s exist for Local mode: `MLXWhisperSTTProvider`
(macOS Apple Silicon, MLX/Metal), `WhisperCppVulkanSTTProvider` (Windows
AMD/Intel, whisper.cpp + Vulkan), and `LocalSTTProvider` (everything else,
faster-whisper -- CUDA on NVIDIA, CPU otherwise). `get_local_provider_kind()`
centralizes the "which local provider" decision in one place (mirroring
`app.core.gpu_probe`'s own "centralize vendor detection once" philosophy),
and `get_local_provider_class()` is a thin dispatch on top of it. The
factory keeps the rest of the codebase -- `STTProvider` contract, cache
layer, router endpoints -- agnostic of which concrete class is in play.

No third-party imports at module level: `mlx_whisper`/`faster_whisper` (and
the httpx-dependent `local_vulkan` module) are pulled in only when the
factory returns the corresponding class, so this module is safe to import
on every platform regardless of which extras are installed.
"""

import os
from enum import Enum
from typing import TYPE_CHECKING

from app.stt.base import STTProvider

if TYPE_CHECKING:
    from app.core.gpu_probe import GpuVendor


class LocalProviderKind(str, Enum):
    APPLE_MLX = "apple_mlx"
    FASTER_WHISPER = "faster_whisper"
    WHISPER_CPP_VULKAN = "whisper_cpp_vulkan"


def is_macos_arm64() -> bool:
    """True only when running natively on Apple Silicon.

    `sys.platform == "darwin" and platform.machine() == "arm64"` excludes
    Rosetta-x86 Python (where `machine()` reports `"x86_64"`); macOS Intel
    therefore falls back to the faster-whisper CPU path.
    """
    import sys

    if sys.platform != "darwin":
        return False
    import platform

    return platform.machine() == "arm64"


def get_local_provider_kind(vendor: "GpuVendor | None" = None) -> LocalProviderKind:
    """Resolve which local STT provider kind applies to this machine.

    Routing rule: macOS arm64 -> `APPLE_MLX` (unchanged, wins regardless of
    `os.name`/vendor); Windows + AMD/Intel GPU -> `WHISPER_CPP_VULKAN`;
    everything else (Windows NVIDIA/none, and non-Windows entirely --
    Linux/macOS-Intel are not supported Local-mode target platforms per
    CLAUDE.md) -> `FASTER_WHISPER`.

    `vendor`: an already-resolved `GpuVendor`, for callers that have already
    paid for a `probe_gpu()` call this cycle (e.g. `local_setup.check_status()`,
    which already calls `_detect_gpu()`) -- skips this function's own
    `probe_gpu()` call so the same uncached, already-expensive probe
    (`docs/TODO.md` -> Tech Debt) doesn't run twice per invocation. Every
    pre-existing caller omits it (default `None`), preserving the original
    self-probing behavior unchanged. This is the single source of truth for
    the AMD/Intel-on-Windows routing rule -- callers that already have a
    vendor must pass it through here rather than re-deriving the rule
    themselves, so the two never drift apart.
    """
    if is_macos_arm64():
        return LocalProviderKind.APPLE_MLX

    if os.name == "nt":
        from app.core.gpu_probe import GpuVendor

        if vendor is None:
            from app.core.gpu_probe import probe_gpu

            vendor = probe_gpu().vendor
        if vendor in (GpuVendor.AMD, GpuVendor.INTEL):
            return LocalProviderKind.WHISPER_CPP_VULKAN

    return LocalProviderKind.FASTER_WHISPER


def get_local_provider_class() -> type[STTProvider]:
    kind = get_local_provider_kind()
    if kind is LocalProviderKind.APPLE_MLX:
        from app.stt.local_mlx import MLXWhisperSTTProvider

        return MLXWhisperSTTProvider
    if kind is LocalProviderKind.WHISPER_CPP_VULKAN:
        from app.stt.local_vulkan import WhisperCppVulkanSTTProvider

        return WhisperCppVulkanSTTProvider
    from app.stt.local import LocalSTTProvider

    return LocalSTTProvider
