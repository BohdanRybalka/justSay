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

block_cipher = None

# Collect data files that PyInstaller's static analysis tends to miss.
# (groq/google-genai both vendor schemas at runtime; pydantic-settings reads
# .env via dotenv but we don't ship a .env in the bundle.)
datas = []
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
    binaries=[],
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
    console=False,        # background sidecar — no console window on Windows
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
