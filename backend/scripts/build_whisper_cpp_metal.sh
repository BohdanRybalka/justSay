#!/usr/bin/env bash
#
# Builds whisper.cpp's `whisper-server` with the Metal backend enabled and
# installs the binary into backend/vendor/whisper-cpp-metal/ for Local-mode
# dev/testing on macOS Apple Silicon.
#
# Mirrors the shape of the Windows sibling
# backend/scripts/build_whisper_cpp_vulkan.ps1 -- same pinned whisper.cpp
# tag, same "clone into a temp dir, build, install only the binary" flow.
# The two scripts share no executable logic (MSVC/Vulkan SDK discovery has
# no macOS counterpart), but the pinned tag below MUST match the Windows
# script's `$WhisperCppTag`; backend/tests/test_build_definitions.py fails
# the suite if the two ever drift apart.
#
# Prerequisites (not installed by this script):
#   - git
#   - The Xcode Command Line Tools (`xcode-select --install`) -- provides
#     clang and the Metal framework headers.
#   - cmake (`brew install cmake`)
#
# The source checkout and CMake build tree live under $TMPDIR, not under
# this repo: only the final binary is copied in, the build tree is
# disposable. See docs/adr/036-one-whisper-cpp-server-provider-for-both-platforms.md.

set -euo pipefail

WHISPER_CPP_TAG="v1.7.6"
REPO_URL="https://github.com/ggerganov/whisper.cpp.git"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="$REPO_ROOT/backend/vendor/whisper-cpp-metal"

WORK_DIR="${TMPDIR:-/tmp}/justsay-whisper-cpp-metal"
SRC_DIR="$WORK_DIR/whisper.cpp"
BUILD_DIR="$WORK_DIR/build"

step() {
    printf '\n==> %s\n' "$1"
}

step "Checking prerequisites"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: this script builds the macOS/Metal binary and only runs on macOS." >&2
    echo "On Windows use backend/scripts/build_whisper_cpp_vulkan.ps1 instead." >&2
    exit 1
fi

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: Metal acceleration needs Apple Silicon; uname -m reports '$(uname -m)'." >&2
    echo "An Intel Mac (or Rosetta) uses the faster-whisper CPU path and needs no binary." >&2
    exit 1
fi

for tool in git cmake; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' is not on PATH. Install it and re-run." >&2
        exit 1
    fi
done

if ! xcode-select -p >/dev/null 2>&1; then
    echo "ERROR: no active developer directory. Run 'xcode-select --install'." >&2
    exit 1
fi

if [ "${1:-}" = "--clean" ]; then
    step "Removing the existing work tree at $WORK_DIR"
    rm -rf "$WORK_DIR"
fi

step "Fetching whisper.cpp $WHISPER_CPP_TAG"

mkdir -p "$WORK_DIR"
if [ -d "$SRC_DIR/.git" ]; then
    echo "Reusing the existing checkout at $SRC_DIR"
else
    rm -rf "$SRC_DIR"
    git clone --depth 1 --branch "$WHISPER_CPP_TAG" "$REPO_URL" "$SRC_DIR"
fi

step "Configuring the CMake build (Metal, static)"

# -DBUILD_SHARED_LIBS=OFF is the load-bearing flag: whisper.cpp's ggml
# defaults it ON, which would produce @rpath-linked dylibs that break once
# the binary is copied into JustSay.app's Resources. The two GGML_METAL
# flags restate upstream defaults explicitly so a future upstream default
# flip cannot silently ship a CPU-only binary.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_SDL2=OFF -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON

step "Building whisper-server (this can take several minutes)"

cmake --build "$BUILD_DIR" --target whisper-server --config Release -j "$(sysctl -n hw.ncpu)"

BIN_DIR="$BUILD_DIR/bin"
SERVER_BIN="$BIN_DIR/whisper-server"
if [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: the build reported success but whisper-server was not found at '$SERVER_BIN'." >&2
    exit 1
fi

step "Installing the binary into $VENDOR_DIR"

mkdir -p "$VENDOR_DIR"
cp "$SERVER_BIN" "$VENDOR_DIR/"
chmod +x "$VENDOR_DIR/whisper-server"

# -DBUILD_SHARED_LIBS=OFF not taking effect is a hard failure, not something
# to work around here. The build-tree binary would carry an absolute
# build-tree LC_RPATH, and copying the dylibs beside it does NOT make dyld
# find them -- there is no cmake --install step and no @loader_path fixup, so
# the result runs from this clone and breaks the moment it is bundled into
# JustSay.app. Stopping now costs one build; shipping it costs a release.
shopt -s nullglob
emitted_dylibs=("$BIN_DIR"/*.dylib)
shopt -u nullglob
if [ ${#emitted_dylibs[@]} -gt 0 ]; then
    echo "ERROR: the build emitted shared libraries, so -DBUILD_SHARED_LIBS=OFF did not take effect:" >&2
    for dylib in "${emitted_dylibs[@]}"; do
        echo "         $(basename "$dylib")" >&2
    done
    echo "       A binary linked this way only runs from this build tree and would break once" >&2
    echo "       bundled into JustSay.app. Please report this with your cmake version" >&2
    echo "       ($(cmake --version | head -1)) -- the pinned whisper.cpp tag builds static by default." >&2
    exit 1
fi

step "Smoke-checking the installed binary"

# stderr is shown, not swallowed: a dyld failure ("Library not loaded:
# @rpath/...") names its own cause precisely, and guessing on the user's
# behalf would point them at the wrong one. Gatekeeper is only the most
# likely cause, so it is offered as a next step rather than asserted -- its
# own symptom ("Killed: 9") does not reach this stream at all.
if ! "$VENDOR_DIR/whisper-server" --help >/dev/null; then
    echo "ERROR: the installed binary did not run. Any message above this line is from the" >&2
    echo "       binary itself and names the real cause." >&2
    echo "       If there is none, macOS most likely quarantined the fresh build -- check:" >&2
    echo "         xattr -l '$VENDOR_DIR/whisper-server'" >&2
    echo "       and clear it with:" >&2
    echo "         xattr -d com.apple.quarantine '$VENDOR_DIR/whisper-server'" >&2
    exit 1
fi

printf '\nDone. whisper-server installed at:\n  %s\n' "$VENDOR_DIR"
printf 'The GGML model itself is NOT bundled here -- WhisperCppServerSTTProvider\n'
printf 'downloads it lazily on first use, same as the other Local-mode providers.\n'
