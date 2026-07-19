# PyInstaller spec for the JustSay backend sidecar (--onedir mode).
#
# Build with:
#   cd backend
#   pip install pyinstaller
#   pyinstaller --clean --noconfirm build_sidecar.spec
#
# Output:
#   backend/dist/justsay-backend/          — directory with exe + all DLLs
#   backend/dist/justsay-backend/justsay-backend(.exe)
#
# The CI release workflow copies the entire directory to src-tauri/resources/.
# Tauri's bundle.resources embeds it as a subdirectory in the installer, placed
# alongside the main executable at install time (no extraction at runtime).
#
# --onedir is used instead of --onefile specifically for Windows:
# --onefile extracts 50+ files to %TEMP%\MEI<hash> on every launch, which
# Windows Defender / AV products routinely quarantine on unsigned binaries,
# killing the process before it can write a single log line.

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
    # sqlite-vec ships its compiled vec0 extension (vec0.dll / vec0.so /
    # vec0.dylib) as package data next to __init__.py, loaded at runtime via
    # sqlite_vec.load() -> conn.load_extension(loadable_path()). Without
    # this the frozen sidecar's history.py._connect() would fail to load
    # the extension (see ADR 001 / spec 003 --selftest-sqlite-vec CI gate).
    + collect_data_files("sqlite_vec")
)

# soundfile is used lazily (pipeline/utils.py duration detection) with a try-except
# fallback, so a missing DLL degrades gracefully. collect_dynamic_libs returns []
# on Windows because soundfile installs as a single .py file (not a package).
# We try anyway; if empty, pipeline falls back to Gemini for unknown-length audio.
binaries = collect_dynamic_libs("soundfile")

# TEN VAD neural silence gate (spec 033 / docs/adr/019-ten-vad-neural-silence-gate.md).
# Loaded INTO this process via ctypes (not spawned as a child like
# whisper-server), so its natural home is inside the PyInstaller bundle
# itself -- landing at _internal/ten_vad/ten_vad.dll, which travels to users
# inside the already-declared resources/justsay-backend directory. That is
# why this spec needs ZERO Tauri-layer changes: no new bundle.resources
# entry, no placeholder mkdir in package.json, and therefore no repeat of
# spec 018's `tauri:dev` breakage.
#
# CONDITIONAL by design: backend/vendor/ten-vad/ is gitignored and populated
# by backend/scripts/fetch_ten_vad.py. A from-source or CI build that never
# ran the fetch script must still build successfully and degrade to the
# energy guard alone (app.audio.vad.resolve_ten_vad_lib() -> None), rather
# than failing the build. release.yml's Windows leg runs the fetch step
# first and then asserts the DLL landed, so a silently VAD-less Windows
# release is impossible despite this tolerance.
# Anchored to SPECPATH (this .spec file's own directory), never the CWD:
# `pyinstaller backend/build_sidecar.spec` from the repo root must bundle the
# DLL identically to a `cd backend` build, instead of silently producing a
# VAD-less sidecar whose only signal is a print() buried in the build log.
#
# The library filename mirrors app.audio.vad._platform_lib_name() rather than
# hardcoding the Windows name. Today only the Windows DLL is ever fetched
# (plan 033 Cuts: "macOS/Linux TEN VAD shipping" is deferred until macOS
# hardware exists), so the non-Windows branches resolve to nothing and the
# build degrades exactly as it does now -- but whoever implements that Cut
# changes the resolver and the fetch script, and this spec then follows along
# instead of silently producing a VAD-less sidecar.
_ten_vad_dir = Path(SPECPATH) / "vendor" / "ten-vad"
if sys.platform == "win32":
    _ten_vad_lib_name = "ten_vad.dll"
elif sys.platform == "darwin":
    _ten_vad_lib_name = "libten_vad.dylib"
else:
    _ten_vad_lib_name = "libten_vad.so"
_ten_vad_lib = _ten_vad_dir / _ten_vad_lib_name
if _ten_vad_lib.is_file():
    binaries += [(str(_ten_vad_lib), "ten_vad")]
    _ten_vad_license = _ten_vad_dir / "LICENSE"
    if _ten_vad_license.is_file():
        datas += [(str(_ten_vad_license), "ten_vad")]
    print(f"build_sidecar: bundling TEN VAD from {_ten_vad_dir}")
else:
    print(
        f"build_sidecar: {_ten_vad_lib} not found — building WITHOUT the neural VAD "
        "(energy guard only). Run backend/scripts/fetch_ten_vad.py to include it."
    )

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
    # sqlite-vec — loadable SQLite extension for semantic search (spec 003)
    "sqlite_vec",
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
    exclude_binaries=True,  # binaries/datas go into COLLECT below (--onedir)
    name="justsay-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX trips Windows Defender / macOS Gatekeeper
    # CONSOLE subsystem so a direct command-line launch
    #   > dist/justsay-backend/justsay-backend.exe --host 127.0.0.1 --port 9377
    # shows uvicorn/FastAPI output for smoke tests + post-mortem diagnostics.
    # The Tauri Rust spawn path on Windows passes CREATE_NO_WINDOW *and*
    # stderr redirected to sidecar.log, so end users never see a console window.
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
    name="justsay-backend",
)
