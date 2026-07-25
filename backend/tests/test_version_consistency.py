"""Four tracked files each declare an app version; they must agree.

package.json feeds npm/Vite. src-tauri/tauri.conf.json feeds the bundle, the
updater manifest, and the numeric FILEVERSION/PRODUCTVERSION fields of the
Win32 version resource. src-tauri/Cargo.toml feeds the string FileVersion/
ProductVersion fields via CARGO_PKG_VERSION, and nothing in tauri-build ever
overwrites it — see docs/adr/030-three-manifests-one-version-checked-by-a-test.md.
src-tauri/Cargo.lock records the `app` package's own version a fourth time;
cargo regenerates it rather than a human editing it, so it goes stale whenever
a bump happens outside `/release`, and every later `cargo build` then dirties
the working tree with a one-line diff. Spec 046's Stage 6 caught exactly that —
the lock still said 0.9.0 while the guard passed on the other three.
A version bump that misses one file must fail here, not ship silently.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = REPO_ROOT / "package.json"
TAURI_CONF_JSON = REPO_ROOT / "src-tauri" / "tauri.conf.json"
CARGO_TOML = REPO_ROOT / "src-tauri" / "Cargo.toml"
CARGO_LOCK = REPO_ROOT / "src-tauri" / "Cargo.lock"


def _json_version(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["version"]


def _cargo_package_field(path: Path, field: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("[package]")
    next_header = re.search(r"^\[", text[start + 1 :], re.M)
    end = start + 1 + next_header.start() if next_header else len(text)
    package_table = text[start:end]
    match = re.search(rf'^\s*{field}\s*=\s*"([^"]+)"', package_table, re.M)
    assert match, f"no [package] {field} found in {path}"
    return match.group(1)


def _cargo_lock_version(path: Path, package_name: str) -> str:
    """The version Cargo.lock records for `package_name`'s own [[package]] entry.

    The name is read out of Cargo.toml rather than hardcoded, so renaming the
    crate surfaces as a failure here instead of silently skipping the check.
    """
    for block in path.read_text(encoding="utf-8").split("[[package]]"):
        name_match = re.search(r'^\s*name\s*=\s*"([^"]+)"', block, re.M)
        if name_match is None or name_match.group(1) != package_name:
            continue
        version_match = re.search(r'^\s*version\s*=\s*"([^"]+)"', block, re.M)
        assert version_match, f"[[package]] {package_name} has no version in {path}"
        return version_match.group(1)
    raise AssertionError(f"no [[package]] named {package_name!r} in {path}")


def test_manifests_declare_one_identical_version() -> None:
    versions = {
        PACKAGE_JSON: _json_version(PACKAGE_JSON),
        TAURI_CONF_JSON: _json_version(TAURI_CONF_JSON),
        CARGO_TOML: _cargo_package_field(CARGO_TOML, "version"),
        CARGO_LOCK: _cargo_lock_version(CARGO_LOCK, _cargo_package_field(CARGO_TOML, "name")),
    }
    distinct = set(versions.values())
    assert len(distinct) == 1, "manifests disagree on version: " + ", ".join(
        f"{path} declares {version}" for path, version in versions.items()
    )
