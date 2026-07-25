#!/usr/bin/env bash
# Builds justsay-audiotap, the macOS system-audio capture helper (spec 074).
#
# One command for CI and for a developer on a Mac. Deliberately strict: a
# compile error must turn the macOS release job red rather than let a bundle
# ship without the helper, because a tag push is the only place the Swift is
# ever compiled (docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md).
#
# Deployment target and product name come from macos/JustSayAudioTap/Package.swift;
# the arch matches release.yml's aarch64-apple-darwin matrix leg.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="$REPO_ROOT/macos/JustSayAudioTap"
BINARY="$PACKAGE_DIR/.build/release/justsay-audiotap"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: build_macos_audio_tap.sh runs on macOS only (uname -s = $(uname -s))." >&2
  exit 1
fi

swift build --package-path "$PACKAGE_DIR" --configuration release --arch arm64

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: swift build reported success but $BINARY is missing." >&2
  exit 1
fi

echo "OK: $BINARY built ($(wc -c < "$BINARY") bytes)"
