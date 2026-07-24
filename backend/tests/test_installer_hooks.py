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


def _sidecar_call(verb: str, sidecar_name: str) -> re.Pattern[str]:
    return re.compile(
        rf'nsis_tauri_utils::{verb}Process(?:CurrentUser)?\s+"{re.escape(sidecar_name)}"'
    )


def test_both_hooks_stop_the_shell_before_the_sidecar() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    bodies = _macro_bodies(text)
    sidecar_name = _sidecar_image_name()
    shell_call = 'CheckIfAppIsRunning "${MAINBINARYNAME}.exe"'
    kill_sidecar = _sidecar_call("Kill", sidecar_name)

    for hook in ("NSIS_HOOK_PREINSTALL", "NSIS_HOOK_PREUNINSTALL"):
        resolved = _resolve_macro_text(hook, bodies)
        assert shell_call in resolved, f"{hook} never calls {shell_call}"
        kill = kill_sidecar.search(resolved)
        assert kill, f"{hook} never kills {sidecar_name}"
        assert resolved.index(shell_call) < kill.start(), (
            f"{hook} must stop {shell_call} before killing {sidecar_name} — "
            "killing the sidecar first lets the shell's watchdog respawn it mid-copy"
        )


def test_both_hooks_wait_for_the_sidecar_and_abort_if_it_survives() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    bodies = _macro_bodies(text)
    sidecar_name = _sidecar_image_name()
    find_sidecar = _sidecar_call("Find", sidecar_name)

    for hook in ("NSIS_HOOK_PREINSTALL", "NSIS_HOOK_PREUNINSTALL"):
        resolved = _resolve_macro_text(hook, bodies)
        assert find_sidecar.search(resolved), (
            f"{hook} kills {sidecar_name} but never polls for it — the kill is "
            "asynchronous, so proceeding immediately can still hit locked files"
        )
        assert "Abort" in resolved, (
            f"{hook} must abort loudly when {sidecar_name} outlives the wait; "
            "continuing is what produces the half-written install"
        )


def test_sidecar_name_reaches_a_process_call_not_only_the_abort_message() -> None:
    text = _installer_hooks_path().read_text(encoding="utf-8")
    sidecar_name = _sidecar_image_name()
    assert _sidecar_call("Kill", sidecar_name).search(text), (
        f"{sidecar_name} must appear in a Kill call, not merely somewhere in the file"
    )
    assert _sidecar_call("Find", sidecar_name).search(text), (
        f"{sidecar_name} must appear in a Find call, not merely somewhere in the file"
    )
