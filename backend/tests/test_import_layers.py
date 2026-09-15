"""The structural rules spec 076 established, enforced instead of documented.

`docs/style-guide.md` §1a states where a backend module goes, and ADR 044
records why. Prose rots; this file fails.

Seven properties are pinned here:

1. `app.core` is a leaf. Only `config.py` (the composition root) and
   `router.py` (operational endpoints) may import a feature package. Letting a
   third module do it is exactly how `core` previously came to hold the
   transcript store, the user preferences and four routers at once.
2. `app.audio.analysis` imports nothing the frozen PyInstaller sidecar lacks,
   and is imported *from* rather than importing — its own docstring names the
   libraries that would break the packaged build, and ADR 015 depends on it.
   A violation here ships broken; only a tag push would otherwise reveal it.
3. No package acquires a web framework, `fastapi` and `starlette` alike,
   outside the modules each package's exempt set names. Every directory under
   `app/` holding any `.py` file is a key in that allowlist — an `__init__.py`
   is not required, so a PEP 420 namespace package cannot be exempt by being
   forgotten — and no exemption survives the import it covers, nor the package
   it names. The modules sitting directly under `app/` are checked at the same
   time, with `main.py` the one exemption: it is the composition root and
   building the FastAPI app is its job.
4. The set of package-level cycles does not grow, and does not outlive the
   cycles it lists.
5. `app/audio/__init__.py` holds a docstring and nothing else, so reaching
   any module in the package costs only that module. Nothing else means
   nothing else: a lazy `__getattr__` re-export defers the cost rather than
   removing it, and puts the package surface this rule deletes straight back.
6. Importing a pure DSP module does not load the capture stack.
7. An underscore-prefixed attribute is private to its own package (ADR 072).
   The package is the directory the module sits in, so a package's `__init__`
   belongs to that package and not to its parent.
   A sibling module may name it; a module in another package may not, in any
   spelling -- `alias._name`, `app.x.y._name` after a plain `import app.x.y`,
   or `from app.x.y import _name`. An underscore-named *module* imported by a
   sibling of its own package is the arrangement working, not a reach-in. The
   allowlist for it ships empty, and `app/` is the whole scope: a test
   legitimately reaches internals and is deliberately not walked.

Every assertion below was mutation-checked when written. The list below is a
ledger of mutations that were actually run, against the module actually named,
with the number of tests each one reddens:

- a core module made to import a feature package, in the absolute
  (`from app.audio import analysis`) and the relative (`from ..audio import
  analysis`) spelling alike -- one test each
- `import fastapi` planted in `app/audio/analysis.py`, the base DSP module --
  three tests, because that module is a non-exempt file of a
  web-framework-free package, is the module property 2 guards, and sits on
  `app.audio.timeline`'s import path
- `from starlette.requests import Request` planted in the same module -- two
  tests, the same first two
- `app/audio/analysis.py` made to import `app.audio.timeline`, absolutely and
  relatively (`from .timeline import ...`) -- one test each
- a fresh `transcripts <-> pipeline` cycle, and a fictional entry added to the
  known-cycle list -- one test each
- a provider given `HTTPException` -- one test
- a `fastapi` importer added as `app/handlers.py`, directly under `app/`, and
  as `app/newpkg/thing.py` in a directory with no `__init__.py` -- one test
  each
- a package key deleted from the web-framework allowlist, a fictional file
  added to an exempt set, and a fictional package key carrying an empty exempt
  set -- one test each
- a recorder import planted in `app/audio/__init__.py` -- two tests, since it
  both grows the package surface and puts the capture stack back on
  `timeline`'s import path -- and a `__getattr__` re-export of the same, one
  test
- `history._lock` read from `app/pipeline/service.py` through
  `from app.transcripts import history`, and through the relative spelling
  `from ..transcripts import history` -- one test each
- `from app.transcripts.history import _lock` planted in the same module --
  one test, which is what proves the `ImportFrom` arm of the walk exists
- `history._lock` read from `app/pipeline/service.py` through the dotted
  spelling a plain `import app.transcripts.history` binds, written out as
  `app.transcripts.history._lock` -- one test, which is what proves the
  attribute arm resolves a chain rather than a single `Name`
- a fictional entry added to the empty package-private allowlist -- one test
- the walk's attribute arm short-circuited to find nothing -- one test, the
  non-vacuity pin, and the gate deliberately stays green, which is why that
  pin is a separate test
- `history._lock` read from `app/transcripts/store_errors.py`, a sibling in
  the same package -- **zero** tests, the negative control proving the rule is
  not over-broad
- a package-private module `app/transcripts/_helpers.py` added alongside an
  `app/transcripts/_zz_consumer.py` spelling `from app.transcripts import
  _helpers` -- **zero** tests, the second negative control: a sibling naming a
  package-private module of its own package is what the rule permits, and the
  `ImportFrom` arm used to record the package rather than the module and fire
  on it
- a package's own `__init__` naming a sibling's private -- `from app.transcripts
  import history` plus `history._lock` appended to
  `app/transcripts/__init__.py` -- **zero** tests, the third negative control.
  A package's `__init__` *is* that package, so trimming the last segment off
  its dotted name would place it in the parent and report the sibling as a
  cross-package reach-in
- `_zz_secret` added to `app/core/__init__.py` and named from a new
  `app/_zz_root.py` directly under `app/` -- **one** test. Under the same
  trimming both sides came out as the string `app` and the reach-in was
  dropped, so this is the mutation that pins the blind spot rather than the
  false alarm
- `from .. import core, _zz_missing` planted in `app/transcripts/` -- **one**
  test, and the offender it names is `app._zz_missing`. The arm used to
  cross-product resolved bases with aliases where the alias walk zips them, so
  the same line named `app.core._zz_missing`, a reach-in nobody writes, and
  never mentioned the module actually reached

Each list below is an allowlist, not a description: adding an entry is a
deliberate act a reviewer can see in the diff.
"""

from __future__ import annotations

import ast
import functools
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

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
        "starlette",
    },
}

_MUST_NOT_IMPORT_APP_MODULE = {
    "audio/analysis.py": {"app.audio.timeline"},
}

_WEB_FRAMEWORK_ROOTS = frozenset({"fastapi", "starlette"})

_WEB_FRAMEWORK_FREE_PACKAGES = {
    "audio": {"router.py", "dependencies.py"},
    "core": {"router.py", "auth_middleware.py", "error_handler.py"},
    "embeddings": set(),
    "pipeline": {"router.py", "service.py", "upload_validation.py"},
    "preferences": {"router.py"},
    "stt": {"router.py"},
    "transcripts": {"history_router.py", "words_router.py"},
}

_WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT = {"main.py"}

_IMPORT_FREE_PACKAGE_INITS = {"audio"}

_KNOWN_PACKAGE_CYCLES = {
    ("app.core", "app.audio"),
    ("app.core", "app.embeddings"),
    ("app.core", "app.stt"),
    ("app.preferences", "app.stt"),
}

_FEATURE_PACKAGES = {
    "audio",
    "embeddings",
    "pipeline",
    "preferences",
    "stt",
    "transcripts",
}


@functools.cache
def _modules() -> Mapping[str, Path]:
    """Every module under `app/`, mapped to its file.

    Cached, so the view handed back is read-only: five call sites share one
    object for the session and a mutation in any of them would silently change
    the tree every later test sees, which surfaces as order dependence rather
    than as a wrong line.
    """
    found = {}
    for path in sorted(_APP_DIR.rglob("*.py")):
        parts = list(path.relative_to(_APP_DIR.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return MappingProxyType(found)


def _containing_package(path: Path) -> str:
    parts = list(path.relative_to(_APP_DIR.parent).with_suffix("").parts)
    return ".".join(parts[:-1])


def _import_from_names(node: ast.ImportFrom, package: str) -> list[str]:
    """Resolve one `from ... import ...` to absolute dotted names.

    A relative import names the same module as its absolute spelling, so both
    must reach the allowlists below as the same string; otherwise one
    `from ..audio import analysis` walks past every gate in this file.
    """
    if not node.level:
        return [node.module] if node.module else []
    parts = package.split(".") if package else []
    parts = parts[: max(len(parts) - node.level + 1, 0)]
    if node.module:
        return [".".join(parts + node.module.split("."))]
    return [".".join(parts + [alias.name]) for alias in node.names]


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _containing_package(path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.extend(_import_from_names(node, package))
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
    """Every directory under `app/` that holds a Python module at any depth.

    An `__init__.py` is deliberately not required: PEP 420 makes
    `app/newpkg/thing.py` importable without one, so keying on `__init__.py`
    would hand a whole directory the "exempt by being forgotten" pass this
    file exists to remove.
    """
    return sorted(
        path.name
        for path in _APP_DIR.iterdir()
        if path.is_dir() and any(path.rglob("*.py"))
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
    the same object with the gate still green.

    The modules sitting directly under `app/` are checked here too. They are
    in no package and were therefore in no allowlist, so `app/handlers.py`
    could hold an endpoint and stay green. `main.py` is the single exemption:
    it is the composition root, and building the FastAPI app is what it is
    for."""
    offenders = []
    for package, exempt in _WEB_FRAMEWORK_FREE_PACKAGES.items():
        package_dir = _APP_DIR / package
        for path in sorted(package_dir.rglob("*.py")):
            if path.relative_to(package_dir).as_posix() in exempt:
                continue
            if _imports_a_web_framework(path):
                offenders.append(path.relative_to(_APP_DIR).as_posix())

    for path in sorted(_APP_DIR.glob("*.py")):
        if path.name in _WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT:
            continue
        if _imports_a_web_framework(path):
            offenders.append(path.name)

    assert not offenders, (
        f"These modules import {sorted(_WEB_FRAMEWORK_ROOTS)}: {offenders}. Raise "
        "a plain exception and let the router map it, per "
        "docs/style-guide.md §3.2."
    )


def test_every_backend_package_is_covered_by_the_web_framework_allowlist():
    """The gate above only sees the packages named in the dict, so an
    unlisted package is exempt in full rather than checked with exceptions.
    That is the defect spec 104 opened on: `core` and `audio` broke the rule
    for as long as they were absent from it. Every directory under `app/` that
    holds a Python module is a key here, with an explicit exempt set — empty
    when the package holds no HTTP-facing module. The modules directly under
    `app/` are covered by the gate itself, not by this dict."""
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
    pass to whatever next takes that path.

    A package key outlives its package the same way, and does it more quietly:
    `rglob` on a directory that no longer exists yields nothing, so the gate
    above stays green while a whole key describes nothing. A key with an empty
    exempt set has no other check on it at all."""
    stale = []
    for package, exempt in sorted(_WEB_FRAMEWORK_FREE_PACKAGES.items()):
        package_dir = _APP_DIR / package
        if not package_dir.is_dir():
            stale.append(f"{package}: no such package")
            continue
        for relative in sorted(exempt):
            path = package_dir / relative
            if not path.exists():
                stale.append(f"{package}/{relative}: no such module")
            elif not _imports_a_web_framework(path):
                stale.append(f"{package}/{relative}: imports no web framework")

    for name in sorted(_WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT):
        path = _APP_DIR / name
        if not path.exists():
            stale.append(f"{name}: no such module")
        elif not _imports_a_web_framework(path):
            stale.append(f"{name}: imports no web framework")

    assert not stale, (
        f"These exemptions no longer cover anything: {stale}. Remove each from "
        "_WEB_FRAMEWORK_FREE_PACKAGES or _WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT — "
        "an exemption that outlives its import silently exempts the next module "
        "to take that path, and a package key that outlives its package checks "
        "nothing while looking like it does."
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


_PACKAGE_PRIVATE_REACH_IN_ALLOWED: dict[tuple[str, str], set[str]] = {}

_LIVE_PACKAGE_PRIVATE_REACH_INS = {
    ("app.transcripts.words", "app.transcripts.history", "_lock"),
    ("app.transcripts.schema", "app.transcripts.vector_store", "_DDL_V3"),
    ("app.transcripts.relocation", "app.transcripts.history", "_conn"),
}


def _module_aliases(tree: ast.Module, package: str, modules: Mapping[str, Path]) -> dict[str, str]:
    """Every local name bound to an `app` module, mapped to that module.

    `_import_from_names` does the relative-spelling work, so `from ..transcripts
    import history` reaches the gate as the same string `from app.transcripts
    import history` does.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            resolved = _import_from_names(node, package)
            if node.level and not node.module:
                for alias, full in zip(node.names, resolved):
                    if full in modules:
                        aliases[alias.asname or alias.name] = full
                continue
            for base in resolved:
                for alias in node.names:
                    full = f"{base}.{alias.name}"
                    if full in modules:
                        aliases[alias.asname or alias.name] = full
    return aliases


def _dotted_name(node: ast.expr) -> str | None:
    """The dotted source spelling of an attribute chain, or None if it is not one.

    `app.transcripts.history` arrives as three nested `Attribute` nodes over a
    `Name`, so a check that only accepts `isinstance(node.value, ast.Name)` sees
    `alias._x` and misses `app.transcripts.history._x` -- which is what a plain
    `import app.transcripts.history` actually binds.
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _resolved_module(dotted: str, aliases: dict[str, str], modules: dict[str, Path]) -> str | None:
    head, _, rest = dotted.partition(".")
    if head in aliases:
        candidate = f"{aliases[head]}.{rest}" if rest else aliases[head]
        if candidate in modules:
            return candidate
    return dotted if dotted in modules else None


def _is_package_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _underscore_reach_ins(
    module: str, path: Path, modules: Mapping[str, Path]
) -> list[tuple[str, str, str]]:
    """Every underscore-prefixed name this module takes from another, as
    `(module, target module, attribute)`.

    Two spellings are walked and both are needed. An attribute access through a
    name bound to an `app` module -- reads and assignments alike, since
    `relocate` writes `history._output_dir` -- and a direct `from app.x.y import
    _name`. Covering only the first would leave the rule one import line away
    from irrelevance, exactly as matching the literal `fastapi` did for rule 3.
    The attribute arm resolves a whole dotted chain, so the alias spelling
    `history._lock` and the plain-import spelling `app.transcripts.history._lock`
    are both seen.

    When `from app.x import _y` names a *module* rather than an attribute, the
    target recorded is that module, not the package it was imported from.
    Recording the package would make a sibling importing a package-private
    module of its own package read as a cross-package reach-in.

    `ast.walk` reaches a function-body import as well as a module-top one, which
    is what makes `app/transcripts/schema.py`'s deferred `vector_store` import
    visible here. Dunders are skipped: `__name__` is not anyone's private state.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _containing_package(path)
    aliases = _module_aliases(tree, package, modules)
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_package_private(node.attr):
            dotted = _dotted_name(node.value)
            target = (
                _resolved_module(dotted, aliases, modules) if dotted is not None else None
            )
            if target is not None:
                found.append((module, target, node.attr))
        elif isinstance(node, ast.ImportFrom):
            resolved = _import_from_names(node, package)
            if node.level and not node.module:
                candidates = list(zip(node.names, resolved))
            else:
                candidates = [
                    (alias, f"{base}.{alias.name}") for base in resolved for alias in node.names
                ]
            for alias, full in candidates:
                if not _is_package_private(alias.name):
                    continue
                base = full.rsplit(".", 1)[0]
                if base not in modules:
                    continue
                found.append((module, full if full in modules else base, alias.name))
    return found


@functools.cache
def _all_underscore_reach_ins() -> frozenset[tuple[str, str, str]]:
    modules = _modules()
    found: set[tuple[str, str, str]] = set()
    for module, path in modules.items():
        found.update(_underscore_reach_ins(module, path, modules))
    return frozenset(found)


@functools.cache
def _module_packages() -> Mapping[str, str]:
    """Each module mapped to the package it lives in, taken from its path.

    Trimming the last dotted segment off a module name is not the same thing
    and gets two shapes wrong. A package's `__init__` *is* that package, so
    `app.transcripts` would come out as living in `app` and a sibling it names
    would read as a cross-package reach-in; and a module directly under `app/`
    would share the string `app` with every package `__init__`, which makes a
    real reach-in through one of them invisible. The directory the file sits in
    answers both.
    """
    return MappingProxyType(
        {module: _containing_package(path) for module, path in _modules().items()}
    )


def test_an_underscore_attribute_is_private_to_its_own_package():
    """ADR 072: the underscore is the only lexical marker Python has for
    package-private, so a sibling in the same package may name it and nothing
    outside may. `app.transcripts.words` reading `history._lock` is the
    arrangement working; `app.pipeline.service` reading it would be a
    cross-package consumer of internals, which means either the name should be
    public or the module is in the wrong package.

    Scope is `app/` only. `backend/tests/**` is deliberately not covered: a test
    legitimately reaches internals to set up state, which
    `tests/test_preferences_router.py` does on purpose."""
    offenders = sorted(
        f"{module} -> {target}.{attribute}"
        for module, target, attribute in _all_underscore_reach_ins()
        if _module_packages()[module] != _module_packages()[target]
        and attribute
        not in _PACKAGE_PRIVATE_REACH_IN_ALLOWED.get((module, target), frozenset())
    )

    assert not offenders, (
        f"These modules name another package's private attribute: {offenders}. "
        "Either the name is part of that module's contract and should lose its "
        "underscore, or the consumer belongs in that package (ADR 072). Adding "
        "a pair to _PACKAGE_PRIVATE_REACH_IN_ALLOWED is a decision, not a "
        "formality -- it ships empty."
    )


def test_no_package_private_exemption_outlives_the_reach_in_it_covers():
    """The mirror ADR 072 requires, in the shape of
    `test_no_web_framework_exemption_outlives_the_import_it_covers`. An entry
    whose module has been deleted, or whose named reach-in is no longer written,
    hands a free pass to whatever next takes that path."""
    modules = _modules()
    live = _all_underscore_reach_ins()
    stale = []
    for (module, target), attributes in sorted(_PACKAGE_PRIVATE_REACH_IN_ALLOWED.items()):
        if module not in modules:
            stale.append(f"{module}: no such module")
            continue
        if target not in modules:
            stale.append(f"{target}: no such module")
            continue
        for attribute in sorted(attributes):
            if (module, target, attribute) not in live:
                stale.append(f"{module} -> {target}.{attribute}: no longer written")

    assert not stale, (
        f"These exemptions no longer cover anything: {stale}. Remove each from "
        "_PACKAGE_PRIVATE_REACH_IN_ALLOWED -- an exemption that outlives its "
        "reach-in silently exempts the next module to take that path."
    )


def test_the_package_private_walk_finds_the_reach_ins_that_are_there():
    """Both tests above pass vacuously on a walk that returns nothing: the gate
    has no offender to report and the mirror has an empty allowlist to check.
    Pin that the walk sees what it is meant to, in the shape of
    `tests/test_naming_rules.py`'s own non-vacuity check.

    Membership, not a count. A count would redden on the next legitimate
    intra-package reach-in, which is the failure direction that gets a gate
    deleted rather than fixed. The three pinned here cover one read of a
    sibling's lock, one attribute reached through a function-body import, and
    one *assignment* into another module's global."""
    missing = sorted(_LIVE_PACKAGE_PRIVATE_REACH_INS - _all_underscore_reach_ins())

    assert not missing, (
        f"The walk no longer finds these reach-ins: {missing}. Either the code "
        "moved -- repoint this set at reach-ins that are actually written -- or "
        "the walk stopped seeing a spelling it used to see, which makes the two "
        "tests above pass while checking nothing."
    )
