"""The Windows NSIS installer hook must stay wired to `tauri.windows.conf.json`
and its process kill list must stay in sync with the sidecar image name the
Rust shell actually uses.

Reads only committed repo files, so it runs on `ubuntu-latest` in CI
(`.github/workflows/ci.yml:114`) the same as on Windows — see ADR 031 and
specs/048-updating-over-running-backend/plan.md.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAURI_WINDOWS_CONF = REPO_ROOT / "src-tauri" / "tauri.windows.conf.json"
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"


def _installer_hooks_path() -> Path:
    config = json.loads(TAURI_WINDOWS_CONF.read_text(encoding="utf-8"))
    relative = config["bundle"]["windows"]["nsis"]["installerHooks"]
    return (REPO_ROOT / "src-tauri" / relative).resolve()


def _macro_bodies(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"!macro\s+(\S+).*?\n(.*?)!macroend", text, re.S)
    }


def _resolve_macro_text(name: str, bodies: dict[str, str], seen: set[str] | None = None) -> str:
    if seen is None:
        seen = set()
    if name not in bodies or name in seen:
        return ""
    seen.add(name)
    resolved = bodies[name]
    for called in re.findall(r"!insertmacro\s+(\S+)", bodies[name]):
        resolved += "\n" + _resolve_macro_text(called, bodies, seen)
    return resolved


def _sidecar_image_name() -> str:
    text = BACKEND_RS.read_text(encoding="utf-8")
    match = re.search(r'if cfg!\(windows\)\s*\{\s*"([^"]+\.exe)"', text)
    assert match, "could not find the Windows sidecar image name literal in backend.rs"
    return match.group(1)


def test_installer_hooks_key_resolves_to_an_existing_file() -> None:
    assert _installer_hooks_path().is_file()


def test_installer_hooks_file_defines_both_hooks() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    bodies = _macro_bodies(text)
    assert "NSIS_HOOK_PREINSTALL" in bodies
    assert "NSIS_HOOK_PREUNINSTALL" in bodies


def test_both_hooks_stop_the_shell_before_the_sidecar() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    bodies = _macro_bodies(text)
    sidecar_name = _sidecar_image_name()
    shell_call = 'CheckIfAppIsRunning "${MAINBINARYNAME}.exe"'
    sidecar_call = f'CheckIfAppIsRunning "{sidecar_name}"'

    for hook in ("NSIS_HOOK_PREINSTALL", "NSIS_HOOK_PREUNINSTALL"):
        resolved = _resolve_macro_text(hook, bodies)
        assert shell_call in resolved, f"{hook} never calls {shell_call}"
        assert sidecar_call in resolved, f"{hook} never calls {sidecar_call}"
        shell_index = resolved.index(shell_call)
        sidecar_index = resolved.index(sidecar_call)
        assert shell_index < sidecar_index, (
            f"{hook} must call {shell_call} before {sidecar_call} — "
            "killing the sidecar first lets the shell's watchdog respawn it mid-copy"
        )


def test_sidecar_name_in_hook_matches_backend_rs() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    sidecar_name = _sidecar_image_name()
    assert sidecar_name in text
