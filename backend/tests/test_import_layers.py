"""The structural rules spec 076 established, enforced instead of documented.

`docs/style-guide.md` §1a states where a backend module goes, and ADR 044
records why. Prose rots; this file fails.

Six properties are pinned here:

1. `app.core` is a leaf. Only `config.py` (the composition root) and
   `router.py` (operational endpoints) may import a feature package. Letting a
   third module do it is exactly how `core` previously came to hold the
   transcript store, the user preferences and four routers at once.
2. `app.audio.analysis` imports nothing the frozen PyInstaller sidecar lacks,
   and is imported *from* rather than importing — its own docstring names the
   libraries that would break the packaged build, and ADR 015 depends on it.
   A violation here ships broken; only a tag push would otherwise reveal it.
3. No package acquires a web framework, `fastapi` and `starlette` alike,
   outside the modules each package's exempt set names. Every package
   directory under `app/` is a key in that allowlist, so a package cannot be
   exempt by being forgotten, and no exemption survives the import it covers.
4. The set of package-level cycles does not grow, and does not outlive the
   cycles it lists.
5. `app/audio/__init__.py` holds a docstring and nothing else, so reaching
   any module in the package costs only that module. Nothing else means
   nothing else: a lazy `__getattr__` re-export defers the cost rather than
   removing it, and puts the package surface this rule deletes straight back.
6. Importing a pure DSP module does not load the capture stack.

Every assertion below was mutation-checked when written: a core module made to
import a feature package, `import fastapi` planted in the base DSP module, the
analysis/timeline direction reversed, a fresh `transcripts <-> pipeline` cycle,
a provider given `HTTPException`, `starlette.requests.Request` planted in the
base DSP module, a package key deleted from the web-framework allowlist, a
fictional file added to an exempt set, a fictional entry added to the
known-cycle list, a recorder import planted in `app/audio/__init__.py`, and a
`__getattr__` re-export planted in the same file. Two of those redden two tests
rather than one, because two properties genuinely overlap on them: `fastapi` in
`app/audio/timeline.py` is both a web framework in a free package and a module
the DSP import must not load, and a recorder import in `app/audio/__init__.py`
both grows the package surface and puts the capture stack back on `timeline`'s
import path.

Each list below is an allowlist, not a description: adding an entry is a
deliberate act a reviewer can see in the diff.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from tests.conftest import assert_import_loads_no_module

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

_WEB_FRAMEWORK_ROOTS = frozenset({"fastapi", "starlette"})

_WEB_FRAMEWORK_FREE_PACKAGES = {
    "audio": {"router.py", "dependencies.py"},
    "core": {"router.py", "auth_middleware.py"},
    "embeddings": set(),
    "llm": set(),
    "pipeline": {"router.py", "service.py", "upload_validation.py"},
    "preferences": {"router.py"},
    "stt": {"router.py"},
    "transcripts": {"history_router.py", "words_router.py", "store_errors.py"},
}

_IMPORT_FREE_PACKAGE_INITS = {"audio"}

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


def _package_directories() -> list[str]:
    return sorted(
        path.name
        for path in _APP_DIR.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )


def _imports_a_web_framework(path: Path) -> bool:
    return any(
        name.split(".")[0] in _WEB_FRAMEWORK_ROOTS for name in _imported_names(path)
    )


def test_providers_do_not_acquire_a_web_framework():
    """A provider executes the Audio-In/Text-Out contract; it has no business
    knowing about HTTP. Keeping the web framework out of these packages is also
    what lets the STT modules import cleanly in the lint job, which installs no
    audio extra.

    Both `fastapi` and `starlette` count. `fastapi.Request` *is*
    `starlette.requests.Request`, re-exported, so a check that matched the
    literal name `fastapi` alone left every module one import line away from
    the same object with the gate still green."""
    offenders = []
    for package, exempt in _WEB_FRAMEWORK_FREE_PACKAGES.items():
        package_dir = _APP_DIR / package
        for path in sorted(package_dir.rglob("*.py")):
            if path.relative_to(package_dir).as_posix() in exempt:
                continue
            if _imports_a_web_framework(path):
                offenders.append(path.relative_to(_APP_DIR).as_posix())

    assert not offenders, (
        f"These modules import {sorted(_WEB_FRAMEWORK_ROOTS)}: {offenders}. Raise "
        "a plain exception and let the router map it, per "
        "docs/style-guide.md §3.2."
    )


def test_every_backend_package_is_covered_by_the_web_framework_allowlist():
    """The gate above only sees the packages named in the dict, so an
    unlisted package is exempt in full rather than checked with exceptions.
    That is the defect spec 104 opened on: `core` and `audio` broke the rule
    for as long as they were absent from it. Every package directory under
    `app/` is a key here, with an explicit exempt set — empty when the package
    holds no HTTP-facing module. `app/main.py` is composition, not a package,
    and is not covered."""
    missing = [
        package
        for package in _package_directories()
        if package not in _WEB_FRAMEWORK_FREE_PACKAGES
    ]

    assert not missing, (
        f"These packages are in no allowlist, so nothing checks them: {missing}. "
        "Add each one to _WEB_FRAMEWORK_FREE_PACKAGES — with an empty exempt "
        "set if it imports no web framework, or with the package-relative path "
        "of every module that legitimately does."
    )


def test_no_web_framework_exemption_outlives_the_import_it_covers():
    """The mirror of `test_the_known_cycle_list_does_not_outlive_the_cycles`,
    for the other allowlist in this file. An exemption whose module has been
    deleted, or which has since dropped its web-framework import, hands a free
    pass to whatever next takes that path."""
    stale = []
    for package, exempt in sorted(_WEB_FRAMEWORK_FREE_PACKAGES.items()):
        package_dir = _APP_DIR / package
        for relative in sorted(exempt):
            path = package_dir / relative
            if not path.exists():
                stale.append(f"{package}/{relative}: no such module")
            elif not _imports_a_web_framework(path):
                stale.append(f"{package}/{relative}: imports no web framework")

    assert not stale, (
        f"These exemptions no longer cover anything: {stale}. Remove each from "
        "_WEB_FRAMEWORK_FREE_PACKAGES — an exemption that outlives its import "
        "silently exempts the next module to take that path."
    )


def _statement_description(node: ast.stmt) -> str:
    if isinstance(node, ast.Import):
        return "import " + ", ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = "." * node.level + (node.module or "")
        names = ", ".join(alias.name for alias in node.names)
        return f"from {module} import {names}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {node.name}"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return type(node).__name__


def test_the_audio_package_surface_holds_nothing_but_a_docstring():
    """A package `__init__.py` executes on every `app.<package>.<module>`
    import, so anything it holds is paid for by every consumer. `app.audio`
    once re-exported both recorders, which made the pure numpy module
    `app.audio.timeline` drag the whole capture stack behind it.

    Checked as "nothing but a docstring" rather than "no import statements",
    because a module-level `__getattr__` restores the same re-export while
    leaving the import statements absent — it defers the cost to the first
    attribute read instead of removing it, and the runtime test below cannot
    see it either, since importing a submodule never invokes it."""
    offenders = []
    for package in sorted(_IMPORT_FREE_PACKAGE_INITS):
        path = _APP_DIR / package / "__init__.py"
        assert path.exists(), (
            f"{package}/__init__.py no longer exists — deleting it turns "
            f"{package} into a namespace package, which changes the pinned "
            "property rather than satisfying it. Update this test."
        )
        body = list(ast.parse(path.read_text(encoding="utf-8")).body)
        if body and ast.get_docstring(ast.Module(body=body, type_ignores=[])):
            body = body[1:]
        for node in body:
            offenders.append(f"{package}/__init__.py: {_statement_description(node)}")

    assert not offenders, (
        f"These package surfaces hold more than a docstring: {offenders}. The "
        "packages listed in _IMPORT_FREE_PACKAGE_INITS pay their __init__.py "
        "cost on every consumer's import, so theirs holds a docstring and "
        "nothing else — not an import, not a lazy __getattr__; import the "
        "submodule directly instead. This is not a project-wide rule: "
        "`app/stt` and `app/embeddings` deliberately re-export from theirs and "
        "are deliberately absent from that set. See docs/style-guide.md §1a."
    )


def test_importing_a_dsp_module_does_not_load_the_capture_stack():
    """The static check above cannot see a transitive acquisition — a recorder
    import appearing in `app/audio/analysis.py` or `app/audio/config.py` would
    cost `timeline` the same 133 modules with `__init__.py` still empty."""
    assert_import_loads_no_module(
        "app.audio.timeline",
        (
            "fastapi",
            "sounddevice",
            "app.audio.recorder",
            "app.audio.meeting_recorder",
        ),
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
