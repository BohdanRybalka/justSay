"""A capability grant the webview cannot reach is a grant nobody asked for.

A capability constrains the webview's IPC calls and nothing else, so a plugin
permission is live only while that plugin's JavaScript binding is installed
(ADR 085). Everything here is read as text out of committed files, so it runs
wherever the suite runs and needs no Tauri toolchain.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_DIR = REPO_ROOT / "src-tauri" / "capabilities"
PACKAGE_JSON = REPO_ROOT / "package.json"

CORE_BINDING = "@tauri-apps/api"


def _granted_permissions() -> tuple[tuple[str, str], ...]:
    """Every permission the capability files grant, as (file name, identifier)."""
    granted: list[tuple[str, str]] = []
    for path in sorted(CAPABILITIES_DIR.glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8"))["permissions"]:
            identifier = entry if isinstance(entry, str) else entry["identifier"]
            granted.append((path.name, identifier))
    assert granted
    return tuple(granted)


def _required_binding(identifier: str) -> str:
    """The npm package that lets the webview issue the call this permission allows."""
    plugin = identifier.split(":", 1)[0]
    return CORE_BINDING if plugin == "core" else f"@tauri-apps/plugin-{plugin}"


def test_every_granted_permission_has_an_installed_javascript_binding():
    dependencies = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["dependencies"]
    unreachable = sorted(
        {
            f"{identifier} in {file_name} needs {_required_binding(identifier)}"
            for file_name, identifier in _granted_permissions()
            if _required_binding(identifier) not in dependencies
        }
    )
    assert not unreachable, (
        "These capability grants name a plugin whose JavaScript binding is not in "
        f"package.json, so nothing in the webview can reach them: {unreachable}. "
        "Install the binding deliberately or delete the grant."
    )
