"""The structural rules spec 076 established, enforced instead of documented.

`docs/style-guide.md` §1a states where a backend module goes, and ADR 044
records why. Prose rots; this file fails.

Four properties are pinned here:

1. `app.core` is a leaf. Only `config.py` (the composition root) and
   `router.py` (operational endpoints) may import a feature package. Letting a
   third module do it is exactly how `core` previously came to hold the
   transcript store, the user preferences and four routers at once.
2. `app.audio.analysis` imports nothing the frozen PyInstaller sidecar lacks,
   and is imported *from* rather than importing — its own docstring names the
   libraries that would break the packaged build, and ADR 015 depends on it.
   A violation here ships broken; only a tag push would otherwise reveal it.
3. No provider package acquires a web framework.
4. The set of package-level cycles does not grow, and does not outlive the
   cycles it lists.

Every assertion below was mutation-checked when written: a core module made to
import a feature package, `import fastapi` planted in the base DSP module, the
analysis/timeline direction reversed, a fresh `transcripts <-> pipeline` cycle,
a provider given `HTTPException`, and a fictional entry added to the known-cycle
list each turned exactly one test red.

Each list below is an allowlist, not a description: adding an entry is a
deliberate act a reviewer can see in the diff.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

_CORE_MAY_IMPORT_UPWARD_FROM = {
    "config.py",
    "router.py",
}

_SIDECAR_ABSENT_LIBRARIES = {
    "audio/analysis.py": {
        "torch",
        "scipy",
        "webrtcvad",
        "silero_vad",
        "onnxruntime",
        "fastapi",
    },
}

_MUST_NOT_IMPORT_APP_MODULE = {
    "audio/analysis.py": {"app.audio.timeline"},
}

_WEB_FRAMEWORK_FREE_PACKAGES = {
    "stt": {"router.py"},
    "embeddings": set(),
    "transcripts": {"history_router.py", "words_router.py", "store_errors.py"},
}

_KNOWN_PACKAGE_CYCLES = {
    ("app.core", "app.audio"),
    ("app.core", "app.embeddings"),
    ("app.core", "app.llm"),
    ("app.core", "app.stt"),
    ("app.preferences", "app.stt"),
}

_FEATURE_PACKAGES = {
    "audio",
    "embeddings",
    "llm",
    "pipeline",
    "preferences",
    "stt",
    "transcripts",
}


def _modules() -> dict[str, Path]:
    found = {}
    for path in sorted(_APP_DIR.rglob("*.py")):
        parts = list(path.relative_to(_APP_DIR.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def _package_of(module: str) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else module


def test_core_reaches_a_feature_package_only_from_its_two_documented_modules():
    """`app.core` is the layer every package may import, so it must not import
    them back. The two exceptions are named in `app/core/__init__.py`."""
    offenders = []
    for module, path in _modules().items():
        if not module.startswith("app.core"):
            continue
        relative = path.relative_to(_APP_DIR).as_posix().removeprefix("core/")
        if relative in _CORE_MAY_IMPORT_UPWARD_FROM:
            continue
        for imported in _imported_names(path):
            head = imported.split(".")
            if len(head) >= 2 and head[0] == "app" and head[1] in _FEATURE_PACKAGES:
                offenders.append(f"{module} -> {imported}")

    assert not offenders, (
        "These app/core modules import a feature package: "
        f"{offenders}. Either the module belongs outside core (see "
        "docs/style-guide.md §1a) or it is a new documented exception, which "
        "means adding it to _CORE_MAY_IMPORT_UPWARD_FROM here AND to the "
        "app/core/__init__.py docstring."
    )


def test_the_base_dsp_module_imports_nothing_the_sidecar_lacks():
    """ADR 015 rests on `audio/analysis.py` staying inside what the frozen
    PyInstaller sidecar actually ships — numpy, soundfile, sounddevice. Its own
    docstring names the libraries that would break the packaged build. A
    violation ships broken rather than failing here, and only a tag push would
    reveal it."""
    offenders = []
    for relative, forbidden in _SIDECAR_ABSENT_LIBRARIES.items():
        path = _APP_DIR / relative
        assert path.exists(), f"{relative} no longer exists — update this test"
        for imported in _imported_names(path):
            root = imported.split(".")[0]
            if root in forbidden:
                offenders.append(f"{relative} imports {imported}")

    assert not offenders, (
        f"{offenders}. See that module's docstring: these are absent from the "
        "frozen sidecar's venv and importing one breaks the packaged build on "
        "both platforms."
    )


def test_the_base_dsp_module_is_imported_from_rather_than_importing():
    """`analysis.py` sits below `timeline.py`, which imports `soxr`. Shared code
    moves *down* into analysis; reversing the direction would put soxr in the
    module every silence detector reaches, which is why `to_mono` lives where it
    does (fix 084)."""
    offenders = []
    for relative, forbidden in _MUST_NOT_IMPORT_APP_MODULE.items():
        path = _APP_DIR / relative
        for imported in _imported_names(path):
            if imported in forbidden:
                offenders.append(f"{relative} imports {imported}")

    assert not offenders, (
        f"{offenders}. The dependency runs the other way: move the shared "
        "function down into this module instead."
    )


def test_providers_do_not_acquire_a_web_framework():
    """A provider executes the Audio-In/Text-Out contract; it has no business
    knowing about HTTP. Keeping fastapi out of these packages is also what lets
    the STT modules import cleanly in the lint job, which installs no audio
    extra."""
    offenders = []
    for package, exempt in _WEB_FRAMEWORK_FREE_PACKAGES.items():
        package_dir = _APP_DIR / package
        if not package_dir.exists():
            continue
        for path in sorted(package_dir.rglob("*.py")):
            if path.name in exempt:
                continue
            if any(name.split(".")[0] == "fastapi" for name in _imported_names(path)):
                offenders.append(path.relative_to(_APP_DIR).as_posix())

    assert not offenders, (
        f"These modules import fastapi: {offenders}. Raise a plain exception "
        "and let the router map it, per docs/style-guide.md §3.2."
    )


def _package_edges() -> dict[tuple[str, str], list[str]]:
    modules = _modules()
    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    for module, path in modules.items():
        for imported in _imported_names(path):
            if not imported.startswith("app"):
                continue
            target = imported if imported in modules else imported.rsplit(".", 1)[0]
            if target not in modules:
                continue
            source_package, target_package = _package_of(module), _package_of(target)
            if source_package != target_package:
                edges[(source_package, target_package)].append(f"{module} -> {imported}")
    return edges


def test_no_package_cycle_beyond_the_ones_already_accounted_for():
    """Spec 076 removed the cycles it could and named the rest. A new pair here
    means a module was placed where it makes two packages depend on each other,
    which is the defect that made `core` unreadable in the first place."""
    edges = _package_edges()
    cycles = {
        tuple(sorted(pair))
        for pair in edges
        if (pair[1], pair[0]) in edges and pair[0] != "app.main"
    }
    known = {tuple(sorted(pair)) for pair in _KNOWN_PACKAGE_CYCLES}

    new = cycles - known
    assert not new, (
        f"New package cycles: {sorted(new)}. Every remaining cycle is listed in "
        "_KNOWN_PACKAGE_CYCLES with the reason it survives; adding to that list "
        "is a decision, not a formality."
    )


def test_the_known_cycle_list_does_not_outlive_the_cycles():
    """The other direction: a cycle that has been fixed must leave this list,
    or the list stops describing anything and the test above goes slack."""
    edges = _package_edges()
    live = {
        tuple(sorted(pair))
        for pair in edges
        if (pair[1], pair[0]) in edges and pair[0] != "app.main"
    }
    stale = {tuple(sorted(pair)) for pair in _KNOWN_PACKAGE_CYCLES} - live

    assert not stale, (
        f"These cycles no longer exist and should be removed from "
        f"_KNOWN_PACKAGE_CYCLES: {sorted(stale)}"
    )
