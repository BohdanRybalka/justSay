"""The build definitions that decide what each platform's release actually
ships must stay wired to the Python resolver that looks for it at runtime.

Spec 068 exists because they were not: the macOS release installed no local
STT engine at all, and nothing in the suite noticed. Every assertion here
reads only committed repo files, so it runs on `ubuntu-latest` in CI
(`.github/workflows/ci.yml`) exactly as it does on Windows or macOS — there
is no Windows or macOS CI job, and `release.yml` only fires on a tag push.
"""

import json
import re
from pathlib import Path

import pytest

from app.stt.local_whisper_cpp_cmd import VENDOR_DIR_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
TAURI_WINDOWS_CONF = REPO_ROOT / "src-tauri" / "tauri.windows.conf.json"
TAURI_MACOS_CONF = REPO_ROOT / "src-tauri" / "tauri.macos.conf.json"
TAURI_SHARED_CONF = REPO_ROOT / "src-tauri" / "tauri.conf.json"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
PACKAGE_JSON = REPO_ROOT / "package.json"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"
VULKAN_BUILD_SCRIPT = REPO_ROOT / "backend" / "scripts" / "build_whisper_cpp_vulkan.ps1"
METAL_BUILD_SCRIPT = REPO_ROOT / "backend" / "scripts" / "build_whisper_cpp_metal.sh"

PLATFORM_CONFS = {
    "win32": TAURI_WINDOWS_CONF,
    "darwin": TAURI_MACOS_CONF,
}

AUDIO_TAP_SCRIPT = REPO_ROOT / "backend" / "scripts" / "build_macos_audio_tap.sh"
AUDIO_TAP_PACKAGE = REPO_ROOT / "macos" / "JustSayAudioTap" / "Package.swift"

_NON_ENGINE_RESOURCES = {"resources/justsay-backend", "resources/justsay-audiotap"}


def _bundle_resources(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["bundle"]["resources"]


def _release_workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8")


def _step_blocks(text: str) -> list[str]:
    """Split the workflow on `- name:` step boundaries so a step's `if:` gate
    can be attributed to that step and not to a neighbour."""
    parts = re.split(r"\n(?=\s*- name: )", text)
    return parts[1:]


def _step_named(fragment: str) -> str:
    matches = [block for block in _step_blocks(_release_workflow_text()) if fragment in block]
    assert len(matches) == 1, f"expected exactly one release step containing {fragment!r}"
    return matches[0]


@pytest.mark.parametrize("platform,config_path", sorted(PLATFORM_CONFS.items()))
def test_each_platform_config_declares_the_sidecar_and_its_vendor_directory(
    platform, config_path
):
    """Both entries are required in each platform file. Repeating the sidecar
    is deliberate (ADR 011): the override must be correct whether Tauri
    deep-merges the `resources` map or replaces it outright.
    """
    resources = _bundle_resources(config_path)
    vendor_dir = VENDOR_DIR_NAMES[platform]

    assert resources.get("resources/justsay-backend") == "justsay-backend"
    assert resources.get(f"resources/{vendor_dir}") == vendor_dir


def test_the_shared_tauri_config_declares_no_vendor_directory():
    """Naming either vendor directory in the shared config fails the *other*
    platform's `tauri build` with a missing resource source — recorded in
    ADR 011 as a mistake already made once."""
    resources = _bundle_resources(TAURI_SHARED_CONF)

    assert resources == {"resources/justsay-backend": "justsay-backend"}


def test_vendor_dir_names_match_what_the_tauri_configs_declare():
    """The symbolic assertion: the runtime resolver and the bundler must name
    the same directory. Either side changing alone fails here rather than
    shipping a binary the backend never finds."""
    declared = {
        platform: [
            value
            for key, value in _bundle_resources(config_path).items()
            if key not in _NON_ENGINE_RESOURCES
        ]
        for platform, config_path in PLATFORM_CONFS.items()
    }

    assert declared == {platform: [name] for platform, name in VENDOR_DIR_NAMES.items()}


def test_the_two_build_scripts_pin_the_same_whisper_cpp_tag():
    """One engine, two recipes: a drift here means Windows and macOS ship
    different whisper.cpp versions with nothing else catching it."""
    vulkan = re.search(
        r'\$WhisperCppTag\s*=\s*"([^"]+)"', VULKAN_BUILD_SCRIPT.read_text(encoding="utf-8")
    )
    metal = re.search(
        r'WHISPER_CPP_TAG="([^"]+)"', METAL_BUILD_SCRIPT.read_text(encoding="utf-8")
    )

    assert vulkan, "no pinned whisper.cpp tag found in the Vulkan build script"
    assert metal, "no pinned whisper.cpp tag found in the Metal build script"
    assert vulkan.group(1) == metal.group(1)


def test_the_metal_build_script_is_a_strict_bash_script():
    text = METAL_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


@pytest.mark.parametrize(
    "step_fragment,runner_os",
    [
        ("Build whisper.cpp with Vulkan backend", "Windows"),
        ("Copy whisper.cpp-Vulkan binary to Tauri resources", "Windows"),
        ("Build whisper.cpp with Metal backend", "macOS"),
        ("Copy whisper.cpp-Metal binary to Tauri resources", "macOS"),
    ],
)
def test_each_engine_build_step_is_gated_on_its_own_runner(step_fragment, runner_os):
    """An ungated Vulkan step breaks the macOS release and vice versa."""
    block = _step_named(step_fragment)

    assert f"if: runner.os == '{runner_os}'" in block


def test_the_metal_steps_name_the_metal_script_and_vendor_directory():
    build = _step_named("Build whisper.cpp with Metal backend")
    copy = _step_named("Copy whisper.cpp-Metal binary to Tauri resources")

    assert "backend/scripts/build_whisper_cpp_metal.sh" in build
    assert "backend/vendor/whisper-cpp-metal/" in copy
    assert "src-tauri/resources/whisper-cpp-metal" in copy


@pytest.mark.parametrize(
    "step_fragment",
    [
        "Build macOS system-audio tap helper",
        "Copy system-audio tap helper to Tauri resources",
        "Verify system-audio tap helper bundled",
    ],
)
def test_each_audio_tap_step_is_gated_on_the_macos_runner(step_fragment):
    """Spec 074: an ungated Swift build breaks the Windows release outright."""
    block = _step_named(step_fragment)

    assert "if: runner.os == 'macOS'" in block


_FAILURE_SWALLOWING = {
    "continue-on-error": r"continue-on-error",
    "|| fallback": r"\|\|",
    "set +e": r"set \+e",
    "; true": r";\s*true\b",
    "|: no-op pipe": r"\|\s*:\s*$",
}


def _failure_swallowing_constructs(block: str) -> list[str]:
    return [
        name
        for name, pattern in _FAILURE_SWALLOWING.items()
        if re.search(pattern, block, re.MULTILINE)
    ]


@pytest.mark.parametrize(
    "step_fragment",
    [
        "Build macOS system-audio tap helper",
        "Copy system-audio tap helper to Tauri resources",
        "Verify system-audio tap helper bundled",
    ],
)
def test_no_audio_tap_step_can_fail_without_failing_the_release(step_fragment):
    """A tag push is the only place this Swift is ever compiled, so a tolerated
    failure would ship a macOS bundle whose meeting recording is inert.

    The previous version of this test listed the two literal spellings
    `continue-on-error` and `|| true`, and an iteration-1 review disproved it by
    appending `|| echo 'skipping helper'` and watching the suite stay green. So
    the assertion is now on the *shape* — any construct that lets a failing
    command leave the step green — rather than on two spellings of it.
    """
    block = _step_named(step_fragment)

    assert _failure_swallowing_constructs(block) == []


def test_the_audio_tap_steps_run_the_build_script_and_verify_its_output():
    build = _step_named("Build macOS system-audio tap helper")
    verify = _step_named("Verify system-audio tap helper bundled")

    assert "backend/scripts/build_macos_audio_tap.sh" in build
    assert "exit 1" in verify


def test_the_failure_swallowing_detector_recognises_a_tolerated_build():
    """The detector above is the whole guarantee, so it is checked against the
    exact mutation that survived the previous test."""
    tolerated = (
        "- name: Build macOS system-audio tap helper\n"
        "  run: bash backend/scripts/build_macos_audio_tap.sh || echo 'skipping helper'\n"
    )

    assert _failure_swallowing_constructs(tolerated) == ["|| fallback"]


def test_the_audio_tap_build_script_is_a_strict_bash_script():
    text = AUDIO_TAP_SCRIPT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


def test_the_audio_tap_deployment_floor_matches_the_bundle_floor():
    """The helper's Swift deployment target and the app's declared macOS floor
    are the same number in two files; drift means a bundle that launches on a
    Mac its own helper refuses to run on."""
    package = AUDIO_TAP_PACKAGE.read_text(encoding="utf-8")
    shared = json.loads(TAURI_SHARED_CONF.read_text(encoding="utf-8"))
    floor = shared["bundle"]["macOS"]["minimumSystemVersion"]

    assert f'.macOS("{floor}")' in package


def test_the_macos_bundle_declares_the_narrow_audio_capture_permission():
    """"System Audio Recording Only", not screen recording — the whole reason
    ADR 041 chose a Core Audio tap over ScreenCaptureKit. Without the key the
    tap is denied at runtime and meeting recording captures silence."""
    macos = json.loads(TAURI_MACOS_CONF.read_text(encoding="utf-8"))
    plist_name = macos["bundle"]["macOS"]["infoPlist"]
    plist = (TAURI_MACOS_CONF.parent / plist_name).read_text(encoding="utf-8")

    assert "NSAudioCaptureUsageDescription" in plist
    assert "NSScreenCaptureUsageDescription" not in plist


def test_only_the_macos_config_ships_the_audio_tap_helper():
    """Naming it in the shared config would fail the Windows `tauri build` with
    a missing resource source — the mistake ADR 011 already records once."""
    assert (
        _bundle_resources(TAURI_MACOS_CONF).get("resources/justsay-audiotap")
        == "justsay-audiotap"
    )
    assert "resources/justsay-audiotap" not in _bundle_resources(TAURI_WINDOWS_CONF)


def test_the_sidecar_pip_install_line_is_unchanged():
    """The chosen engine is a bundled binary, not a Python package. Adding an
    extra here would pull a multi-GB dependency into the shipped sidecar for
    nothing."""
    assert 'pip install -e ".[cloud,audio]"' in _release_workflow_text()


def test_pyproject_scopes_package_discovery_without_disabling_it():
    """Discovery must be *narrowed* to `app`, never replaced by a literal list.

    Unscoped, setuptools scans the whole directory and finds two top-level
    packages the moment `backend/vendor/` exists -- which it does on any machine
    that has built the local whisper.cpp engine -- and refuses to build. The
    obvious fix, `[tool.setuptools] packages = ["app"]`, is worse than the bug:
    an explicit list does not recurse, so it silently drops every subpackage.
    Measured on a clean worktree, wheel contents: unscoped 54 `app` files and
    all six subpackages; `packages = ["app"]` **2 files and none**;
    `packages.find include = ["app*"]` 54 files again, with `vendor` excluded.

    Editable installs mask the difference, and every install this repo performs
    is editable -- so nothing else here would catch the regression. `release.yml`
    installs this manifest into the PyInstaller environment, which is why it is
    worth a test at all.
    """
    text = PYPROJECT.read_text(encoding="utf-8")

    assert re.search(r"^\[tool\.setuptools\.packages\.find\]$", text, re.MULTILINE), (
        "[tool.setuptools.packages.find] is missing; an unscoped scan picks up vendor/"
    )
    section = re.search(
        r"^\[tool\.setuptools\.packages\.find\]$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert re.search(r'^include = \["app\*"\]$', section.group(1), re.MULTILINE), (
        "include must be the glob 'app*', so subpackages are discovered, not just 'app'"
    )
    assert not re.search(r"^\[tool\.setuptools\]\s*$", text, re.MULTILINE), (
        "a bare [tool.setuptools] table risks a literal `packages` list, which does not recurse"
    )


def test_every_app_subpackage_on_disk_is_covered_by_the_discovery_glob():
    """The glob above is only correct while every package lives under `app/`.

    Pairs with it deliberately: that test pins the declaration, this one pins
    the assumption the declaration rests on, so a package added outside `app/`
    fails here instead of going missing from a built wheel.
    """
    backend_root = PYPROJECT.parent
    roots = {p.relative_to(backend_root).parts[0] for p in backend_root.glob("*/__init__.py")}

    assert roots == {"app", "tests"}, (
        f"top-level Python packages are {sorted(roots)}; the discovery glob only covers 'app*'. "
        "'tests' is expected and never shipped -- setuptools excludes it by default, which is "
        "why an unscoped scan tolerated it and still choked on 'vendor'."
    )


def test_pyproject_declares_no_apple_specific_extra():
    """The macOS engine is a bundled binary, so no extra installs it.

    Spec 068's own AC tried to prove this with `pip install -e ".[local-mac]"`
    failing. It does not fail: pip 23.0.1 warns `does not provide the extra`
    and exits 0, so that check could never have caught a reintroduction. Nor
    would the AC's grep have — its paths cover `backend/app`, `backend/tests`,
    `.github`, `src-tauri` and `package.json`, but not this file, which is the
    one place the extra ever lived. Hence a test that reads it directly.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    declared = set(re.findall(r"^([a-z][a-z-]*) = \[", section, re.MULTILINE))

    assert declared == {"dev", "cloud", "local", "local-llm", "audio"}, (
        f"pyproject extras changed: {sorted(declared)}. An Apple-specific extra "
        "would mean the macOS engine is a Python package again, which spec 068 "
        "and ADR 036 removed."
    )
    assert not re.search(r"\bmlx\b|mlx[-_]whisper", text, re.IGNORECASE)


@pytest.mark.parametrize("vendor_dir", sorted(VENDOR_DIR_NAMES.values()))
def test_the_frontend_build_scripts_create_both_vendor_directories(vendor_dir):
    """Tauri fails the build when a declared resource directory does not
    exist, and each platform config declares only its own — but both scripts
    run on both platforms, so both directories must be created."""
    scripts = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["scripts"]

    for script_name in ("dev", "build"):
        assert vendor_dir in scripts[script_name], (
            f"npm run {script_name} does not create src-tauri/resources/{vendor_dir}"
        )
