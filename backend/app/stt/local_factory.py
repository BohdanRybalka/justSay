"""Platform-aware local STT provider selection.

On macOS Apple Silicon (arm64) the local STT path is `MLXWhisperSTTProvider`
(MLX/Metal acceleration). Everywhere else it stays on `LocalSTTProvider`
(faster-whisper). The factory keeps the rest of the codebase — `STTProvider`
contract, cache layer, router endpoints — agnostic of which concrete class is
in play.

No third-party imports at module level: ``mlx_whisper`` and ``faster_whisper``
are pulled in only when the factory returns the corresponding class, so this
module is safe to import on every platform regardless of which extras are
installed.
"""

from app.stt.base import STTProvider


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


def get_local_provider_class() -> type[STTProvider]:
    if is_macos_arm64():
        from app.stt.local_mlx import MLXWhisperSTTProvider

        return MLXWhisperSTTProvider
    from app.stt.local import LocalSTTProvider

    return LocalSTTProvider
