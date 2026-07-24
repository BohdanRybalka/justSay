"""Three tracked manifests each declare an app version; they must agree.

package.json feeds npm/Vite. src-tauri/tauri.conf.json feeds the bundle, the
updater manifest, and the numeric FILEVERSION/PRODUCTVERSION fields of the
Win32 version resource. src-tauri/Cargo.toml feeds the string FileVersion/
ProductVersion fields via CARGO_PKG_VERSION, and nothing in tauri-build ever
overwrites it — see docs/adr/030-three-manifests-one-version-checked-by-a-test.md.
A version bump that misses one manifest must fail here, not ship silently.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = REPO_ROOT / "package.json"
TAURI_CONF_JSON = REPO_ROOT / "src-tauri" / "tauri.conf.json"
CARGO_TOML = REPO_ROOT / "src-tauri" / "Cargo.toml"


def _json_version(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["version"]


def _cargo_package_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("[package]")
    next_header = re.search(r"^\[", text[start + 1 :], re.M)
    end = start + 1 + next_header.start() if next_header else len(text)
    package_table = text[start:end]
    version_match = re.search(r'^\s*version\s*=\s*"([^"]+)"', package_table, re.M)
    assert version_match, f"no [package] version found in {path}"
    return version_match.group(1)


def test_manifests_declare_one_identical_version() -> None:
    versions = {
        PACKAGE_JSON: _json_version(PACKAGE_JSON),
        TAURI_CONF_JSON: _json_version(TAURI_CONF_JSON),
        CARGO_TOML: _cargo_package_version(CARGO_TOML),
    }
    distinct = set(versions.values())
    assert len(distinct) == 1, "manifests disagree on version: " + ", ".join(
        f"{path} declares {version}" for path, version in versions.items()
    )
