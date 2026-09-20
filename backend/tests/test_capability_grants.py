"""Pins on the plugin permissions a capability may grant the webview.

A capability constrains the webview's IPC calls and never Rust, so the shell
plugin's named-binary scope validated nothing and is gone (ADR 085). The
filesystem grant went the same way once the page stopped reading a dropped file
by path (ADR 087). Deleting the shell scope also removed the last visible sign
that its registration is load-bearing, which the third pin holds. Everything is
read as text out of committed files, so all three run wherever the suite runs.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_DIR = REPO_ROOT / "src-tauri" / "capabilities"
TAURI_SHARED_CONF = REPO_ROOT / "src-tauri" / "tauri.conf.json"
LIB_RS = REPO_ROOT / "src-tauri" / "src" / "lib.rs"

CAPABILITY_SUFFIXES = (".json", ".json5", ".toml")
SHELL_PERMISSION_PREFIX = "shell:"
FS_PERMISSION_PREFIX = "fs:"
SHELL_PLUGIN_REGISTRATION = "tauri_plugin_shell::init()"


def _capability_sources() -> tuple[tuple[str, str], ...]:
    """Every file Tauri reads capabilities from, as (label, text).

    Covers the whole `capabilities/**/*` tree in all three accepted formats,
    plus the inline capabilities `tauri.conf.json` may carry.
    """
    found = [
        (str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8"))
        for path in sorted(CAPABILITIES_DIR.rglob("*"))
        if path.is_file() and path.suffix in CAPABILITY_SUFFIXES
    ]
    found.append(
        (
            str(TAURI_SHARED_CONF.relative_to(REPO_ROOT)),
            TAURI_SHARED_CONF.read_text(encoding="utf-8"),
        )
    )
    assert found
    return tuple(found)


def test_no_capability_source_grants_a_shell_permission():
    granting = sorted(
        label for label, text in _capability_sources() if SHELL_PERMISSION_PREFIX in text
    )

    assert not granting, (
        f"These files grant a shell permission again: {granting}. The shell plugin ships its "
        "own webview binding, so a grant here is reachable with no npm package installed and "
        "no call site in src/. Re-grant it only alongside the call site that needs it."
    )


def test_the_shell_plugin_stays_registered():
    assert SHELL_PLUGIN_REGISTRATION in LIB_RS.read_text(encoding="utf-8"), (
        f"{SHELL_PLUGIN_REGISTRATION} is gone from {LIB_RS.name}. It reads as dead registration "
        "now that no capability grants a shell permission, but it is the only place `Shell` is "
        "managed, so removing it panics every production launch at the sidecar spawn. No Rust "
        "test builds an AppHandle, so nothing else catches this."
    )


def test_no_capability_source_grants_a_filesystem_permission():
    granting = sorted(
        label for label, text in _capability_sources() if FS_PERMISSION_PREFIX in text
    )

    assert not granting, (
        f"These files grant a filesystem permission again: {granting}. Nothing reads a dropped "
        "file by path any more — the page is handed a File object the user chose (ADR 087) — so "
        "a grant here is a webview-reachable read of every listed root with no call site behind "
        "it. Re-grant it only alongside the call site that needs it."
    )
