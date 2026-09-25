"""Pins on what the Rust shell lets a webview reach, read out of committed files.

A capability constrains the webview's IPC calls and never Rust, so the shell
plugin's named-binary scope validated nothing and is gone (ADR 085). The
filesystem grant went the same way once the page stopped reading a dropped file
by path (ADR 087). Deleting the shell scope also removed the last visible sign
that its registration is load-bearing. The widget window, built in Rust rather
than declared in the config, is pinned here because it has no page-side guard.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_DIR = REPO_ROOT / "src-tauri" / "capabilities"
TAURI_SHARED_CONF = REPO_ROOT / "src-tauri" / "tauri.conf.json"
LIB_RS = REPO_ROOT / "src-tauri" / "src" / "lib.rs"
WIDGET_WINDOW_RS = REPO_ROOT / "src-tauri" / "src" / "widget_window.rs"

CAPABILITY_SUFFIXES = (".json", ".json5", ".toml")
PARSEABLE_SUFFIX = ".json"
SHELL_PERMISSION_PREFIX = "shell:"
FS_PERMISSION_PREFIX = "fs:"
SHELL_PLUGIN_REGISTRATION = "tauri_plugin_shell::init()"
DRAG_DROP_OPT_OUT = "drag_drop_enabled"

WIDGET_BUILDER_PATTERN = re.compile(
    r"WebviewWindowBuilder::new\([^;]*?\"widget\"[^;]*?\.build\(\)", re.DOTALL
)


def _permission_identifiers(capability: object) -> list[str]:
    """Every permission identifier one capability grants.

    An entry is either the identifier as a bare string or an object carrying it
    under `identifier` alongside its scope.
    """
    if not isinstance(capability, dict):
        return []
    identifiers = []
    for permission in capability.get("permissions", []):
        if isinstance(permission, str):
            identifiers.append(permission)
        elif isinstance(permission, dict) and isinstance(permission.get("identifier"), str):
            identifiers.append(permission["identifier"])
    return identifiers


def _granted_permissions() -> tuple[tuple[str, str], ...]:
    """Every permission the shipped capabilities grant, as (label, identifier).

    Covers the whole `capabilities/**/*` tree plus the inline capabilities
    `tauri.conf.json` may carry, and refuses a capability format it cannot parse
    rather than reading past it.
    """
    granted: list[tuple[str, str]] = []
    for path in sorted(CAPABILITIES_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in CAPABILITY_SUFFIXES:
            continue
        label = str(path.relative_to(REPO_ROOT))
        assert path.suffix == PARSEABLE_SUFFIX, (
            f"{label} is a capability format this gate has no parser for, so every permission "
            "it grants goes unread. Add the parser before shipping the file."
        )
        capability = json.loads(path.read_text(encoding="utf-8"))
        granted.extend(
            (label, identifier) for identifier in _permission_identifiers(capability)
        )

    conf_label = str(TAURI_SHARED_CONF.relative_to(REPO_ROOT))
    conf = json.loads(TAURI_SHARED_CONF.read_text(encoding="utf-8"))
    for inline in conf.get("app", {}).get("security", {}).get("capabilities", []):
        granted.extend(
            (conf_label, identifier) for identifier in _permission_identifiers(inline)
        )

    assert granted
    return tuple(granted)


def _widget_builder_chain() -> str:
    """The widget window's `WebviewWindowBuilder` chain, as written in its module."""
    match = WIDGET_BUILDER_PATTERN.search(WIDGET_WINDOW_RS.read_text(encoding="utf-8"))
    assert match, (
        f"no widget WebviewWindowBuilder chain found in {WIDGET_WINDOW_RS.name}. The window "
        "moved or was renamed, and this gate now reads nothing."
    )
    return match.group(0)


def test_no_capability_source_grants_a_shell_permission():
    granting = sorted(
        {
            label
            for label, identifier in _granted_permissions()
            if identifier.startswith(SHELL_PERMISSION_PREFIX)
        }
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
        {
            label
            for label, identifier in _granted_permissions()
            if identifier.startswith(FS_PERMISSION_PREFIX)
        }
    )

    assert not granting, (
        f"These files grant a filesystem permission again: {granting}. Nothing reads a dropped "
        "file by path any more — the page is handed a File object the user chose (ADR 087) — so "
        "a grant here is a webview-reachable read of every listed root with no call site behind "
        "it. Re-grant it only alongside the call site that needs it."
    )


def test_the_widget_window_keeps_the_shell_drag_drop_handler():
    assert DRAG_DROP_OPT_OUT not in _widget_builder_chain(), (
        f"the widget window now sets {DRAG_DROP_OPT_OUT}. At false the page receives external "
        "drops, and src/widget/widget.ts registers no drop listener, so a dropped file "
        "navigates a transparent, undecorated, always-on-top webview to a file:// URL the user "
        "cannot leave without restarting the app. Disable it only alongside a page-side guard "
        "like the Settings window's (ADR 087)."
    )
