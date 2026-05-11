# PyInstaller spec for the JustSay backend sidecar.
#
# Build with:
#   cd backend
#   pip install pyinstaller
#   pyinstaller --clean --noconfirm build_sidecar.spec
#
# Output:
#   backend/dist/justsay-backend-<TARGET_TRIPLE>/   (--onedir tree)
#   backend/dist/justsay-backend-<TARGET_TRIPLE>(.exe)
#
# Tauri's `bundle.externalBin` will pick the right per-platform binary based on
# the host triple (e.g. `justsay-backend-x86_64-pc-windows-msvc.exe`,
# `justsay-backend-aarch64-apple-darwin`). Wrap this spec in a build script
# that renames the output to the target triple before `tauri build`.

# ruff: noqa
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect data files that PyInstaller's static analysis tends to miss.
# google-genai ships protobuf descriptors as package data and groq ships
# JSON schema files for tool calling. Both modules use `importlib.resources`
# to load them at runtime, which PyInstaller does not infer statically.
# Without this, the first real /stt/transcribe call against Gemini explodes
# inside the frozen binary with a FileNotFoundError deep in the SDK.
datas = (
    collect_data_files("google.genai")
    + collect_data_files("groq")
)

# soundfile bundles libsndfile as a runtime DLL/dylib that PyInstaller misses
# via static analysis. Without this, every Audio code path that opens a WAV
# raises OSError("sndfile library not found") inside the frozen binary.
binaries = collect_dynamic_libs("soundfile")

hiddenimports = [
    # FastAPI/uvicorn picks these up via dynamic dispatch
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # Audio extras
    "soundfile",
    "sounddevice",
    # numpy is pulled in via sounddevice / soundfile but PyInstaller may miss
    # its C extension hooks under certain Python builds; declare explicitly.
    "numpy",
    # Cloud SDKs
    "groq",
    "google.genai",
    # System monitoring
    "psutil",
]

# faster-whisper / ctranslate2 are large; they ship as ``[local]`` and only
# matter when the user actually clicks Local mode. Exclude from default sidecar
# to keep size sane; a separate `local-sidecar.spec` would build the heavy one
# if/when we wire Plan 006.
excludes = [
    "faster_whisper",
    "ctranslate2",
    "torch",
    "torchaudio",
    "ollama",
]

a = Analysis(
    ["app/main.py"],
    pathex=[str(Path(".").resolve())],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="justsay-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX trips Windows Defender / macOS Gatekeeper
    # CONSOLE subsystem so a direct command-line launch
    #   > dist/justsay-backend/justsay-backend.exe --host 127.0.0.1 --port 9377
    # shows uvicorn/FastAPI output for smoke tests + post-mortem diagnostics.
    # The Tauri Rust spawn path on Windows passes CREATE_NO_WINDOW *and*
    # stdout/stderr=Stdio::null, so end users never see a console window AND
    # uvicorn logs go nowhere — the binary still writes the structured
    # backend.log via logging_config (file handler), so production diagnostics
    # land there, not in stdout. Direct CLI launches are the diagnostic path.
    console=True,
    target_arch=None,
    codesign_identity=None,  # signing happens in the CI release workflow
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="justsay-backend",
)
