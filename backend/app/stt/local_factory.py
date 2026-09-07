"""Platform-aware local STT provider selection.

Two concrete `STTProvider`s exist for Local mode:
`WhisperCppServerSTTProvider` (macOS Apple Silicon via Metal, and Windows
AMD/Intel via Vulkan -- one class, the backend baked into the binary) and
`LocalSTTProvider` (everything else, faster-whisper -- CUDA on NVIDIA, CPU
otherwise). `get_local_provider_kind()` centralizes the "which local
provider" decision in one place (mirroring `app.core.gpu_probe`'s own
"centralize vendor detection once" philosophy), and
`get_local_provider_class()` is a thin dispatch on top of it. The factory
keeps the rest of the codebase -- `STTProvider` contract, cache layer,
router endpoints -- agnostic of which concrete class is in play.

No third-party imports at module level: `faster_whisper` (and the
httpx-dependent `local_whisper_cpp` module) are pulled in only when the
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
    FASTER_WHISPER = "faster_whisper"
    WHISPER_CPP_SERVER = "whisper_cpp_server"


_ACCELERATED_DEVICES: dict[LocalProviderKind, frozenset[str]] = {
    LocalProviderKind.FASTER_WHISPER: frozenset({"cuda"}),
    LocalProviderKind.WHISPER_CPP_SERVER: frozenset({"metal", "vulkan"}),
}


def is_accelerated_device(device: str, kind: LocalProviderKind) -> bool:
    """Whether the provider that will load actually reaches a GPU on `device`.

    The rule keys on the provider because the device string alone does not
    settle it. faster-whisper reaches CTranslate2, which has a CUDA backend
    and no Metal or Vulkan one, so a `whisper_device` hand-set to `"metal"` on
    a machine routed to faster-whisper is an int8 CPU load however it is
    spelled -- reporting a GPU for it describes a path that cannot load.
    whisper.cpp is the opposite: Metal and Vulkan are exactly its fp16 GPU
    backends.

    A kind this mapping does not list -- a third `LocalProviderKind` is
    anticipated by this module's own docstring -- accelerates nothing, so the
    answer degrades to the conservative CPU one instead of raising a `KeyError`
    into `GET /stt/local/status`.
    """
    return device in _ACCELERATED_DEVICES.get(kind, frozenset())


def compute_type_for_device(device: str, kind: LocalProviderKind) -> str:
    """The compute type a device implies for the provider that will load it.

    fp16 is exactly what an accelerated device buys, so this is the same rule
    `is_accelerated_device` states, read as a compute type. Every other device
    -- `"cpu"`, and any unrecognized `whisper_device` the user typed -- gets
    `"int8"`.
    """
    return "float16" if is_accelerated_device(device, kind) else "int8"


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

    Routing rule: macOS arm64 -> `WHISPER_CPP_SERVER` (wins regardless of
    `os.name`/vendor, and involves no GPU probe -- Apple Silicon always has
    Metal); Windows + AMD/Intel GPU -> `WHISPER_CPP_SERVER`; everything else
    (Windows NVIDIA/none, and non-Windows entirely -- Linux/macOS-Intel are
    not supported Local-mode target platforms per CLAUDE.md) ->
    `FASTER_WHISPER`.

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
        return LocalProviderKind.WHISPER_CPP_SERVER

    if os.name == "nt":
        from app.core.gpu_probe import GpuVendor

        if vendor is None:
            from app.core.gpu_probe import probe_gpu

            vendor = probe_gpu().vendor
        if vendor in (GpuVendor.AMD, GpuVendor.INTEL):
            return LocalProviderKind.WHISPER_CPP_SERVER

    return LocalProviderKind.FASTER_WHISPER


def get_local_provider_class() -> type[STTProvider]:
    if get_local_provider_kind() is LocalProviderKind.WHISPER_CPP_SERVER:
        from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider

        return WhisperCppServerSTTProvider
    from app.stt.local import LocalSTTProvider

    return LocalSTTProvider
