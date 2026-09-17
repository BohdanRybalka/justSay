"""The structural rules spec 076 established, enforced instead of documented.

`docs/style-guide.md` §1a states where a backend module goes, and ADR 044
records why. Prose rots; this file fails.

Ten properties are pinned here:

1. No module under `app.core` imports a feature package, with no exception at
   all: `core` is the layer every other package may import, and importing one
   back is how it previously came to hold the transcript store, the user
   preferences and four routers at once. Separately, exactly one module in the
   whole of `app/` imports the composition root `app.config` directly —
   `core/config.py`, which re-exports it so every call site keeps spelling the
   singleton `app.core.config.settings`. That half is scoped to all of `app/`
   rather than to `core`, because the second spelling is plantable in any
   package, and it prefix-matches so `app.config.runtime` could not walk past
   if `app/config.py` ever became `app/config/`. Neither half makes `core` a
   leaf, and nothing here claims one: `app.core.config` re-exports a settings
   object assembled from every feature package, so importing it still loads
   all of them (ADR 076).
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
   time, with `main.py` the one exemption: building the FastAPI application
   is its job. The composition root is `app/config.py`, a different module.
4. The set of *two-node* package cycles does not grow and does not outlive the
   pairs it lists, and — separately — the set of packages that all reach each
   other does not change in either direction. The first instrument sees one
   pair today; nineteen of this graph's twenty elementary cycles run longer
   than two nodes and are invisible to it, which is why the second exists and
   why the tolerance list is named for the scope it actually has. Neither
   number measures progress on the other, and a falling two-node count is not
   evidence that the graph untangled.
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
8. A package `__init__.py` named as re-export-only holds a docstring, imports
   and the `__all__` it publishes, and nothing else in any position. It is an
   allowlist rather than a list of banned node types: a cache and a lock
   wrapped in `if TYPE_CHECKING:` or in `try: … except ImportError:` are
   neither a definition nor a top-level assignment, and a gate that classifies
   what it recognises accepts everything it does not. `app/stt/__init__.py`
   held a routing layer, a provider cache and a `threading.Lock` across 215
   lines, which is why the HTTP router could not take the obvious name and why
   seven call sites imported the package from inside a function body. This is
   the weaker sibling of property 5: the packages listed there hold nothing at
   all, the ones listed here hold a list of names and no behaviour. The
   allowlist covers `stt` alone; `app/embeddings/__init__.py` still holds the
   same shape — two locks, three cache globals, a Protocol and two functions —
   and adding it here would fail rather than pin, because making it pass means
   splitting that package too.
9. Nothing under `app/` defers an import of the `app.stt` package into a
   function body except `app/main.py`, where the deferral buys late binding
   rather than startup time: `import app.main` has already loaded the package
   through `app/main.py:31` before `lifespan()` runs, and three tests replace
   `clear_cache` on the module object the shutdown body reads it from.
   `docs/style-guide.md` reads every other function-local `from app.…` as a
   cycle being hidden; this gate does not enforce that general rule, only the
   `app.stt` package half of it, because that is the package that had seven of
   them.
10. A system-audio capture source reaches neither `soxr` nor the module that
   imports it. Both capture callbacks run on the audio thread and neither has
   any resampling to do; the deinterleave they share lives in `analysis.py`
   beside the `to_mono` it calls. The sources are found by subclass rather
   than by a typed list, so the next platform's is covered the day it is
   written.

Every assertion below was mutation-checked when written. The list below is a
ledger of mutations that were actually run, against the module actually named,
with the number of tests each one reddens:

- a core module made to import a feature package, in the absolute
  (`from app.audio import analysis`) and the relative (`from ..audio import
  analysis`) spelling alike -- one test each
- `from app.audio.config import AudioSettings` planted in `app/core/router.py`,
  which held that permission until spec 165 emptied the list -- **two** tests,
  not one: the feature-package rule, and the two-node cycle test below, because
  `app.core <-> app.audio` left `_KNOWN_TWO_NODE_PACKAGE_CYCLES` in the same
  change and so is a new pair again. On `99e05e4` that same line reddened
  **zero**
- `from app.config import settings` planted in `app/transcripts/search.py` in
  place of its function-local `app.core.config` import, so the module still
  works and only the spelling reaches past the one doorway -- **one** test.
  It is planted outside `core` on purpose: the gate walked `app.core` alone
  when it shipped, so this exact line was green in seven packages out of eight
- `from app.config.runtime import settings` planted in `app/core/router.py`'s
  `_raise_stop_signal` -- **one** test. The submodule does not exist and is not
  meant to; what it pins is that the upward check prefix-matches rather than
  comparing for equality, which is how it shipped
- both of the above in one diff -- `from app.audio.config import AudioSettings`
  added to `app/core/router.py` while its `app.core.config` import becomes
  `app.config` -- **three** tests, and the point of the mutation is that the
  feature-package rule and the composition-root rule report *both* messages.
  They were one test function with two asserts until Stage 5, where the first
  `assert` firing hid the second
- `app/config.py` renamed to `app/composition.py` with the re-export repointed
  -- **two** tests of this file, the existence mirror and the component pin.
  Measured against this file alone: the full suite stops at collection, because
  `tests/test_settings_isolation.py` imports the composition root by name.
  Before the mirror existed the same rename left all fourteen tests here green,
  with the gate matching nothing at all
- a function-local `from app.pipeline import service` added to
  `app/stt/base.py` -- **two** tests, the component pin (seven names to eight)
  and the two-node list, since `app.pipeline <-> app.stt` is also a new pair.
  The same edge takes the elementary-cycle enumeration from 20 to 44, which is
  why membership is pinned and the enumeration is not
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
- `from app.audio import timeline` planted in `app/audio/windows_loopback.py`
  -- **one** test. That spelling used to arrive at the allowlists as
  `app.audio` and walk past every rule written at module granularity, this one
  included; the resolver now returns the submodule alongside the package
- `import soxr` planted in `app/audio/macos_tap.py` -- **one** test
- `import soxr` planted in `app/audio/system_source.py`, a sibling both capture
  sources import, so no capture module spells it anywhere -- **one** test, and
  only the runtime probe sees it. The static walk reports nothing, which is the
  blind spot the probe is there for
- `_get_or_create` moved back into `app/stt/__init__.py`, and separately
  `_cache_lock = threading.Lock()` put back beside it -- one test each, the
  re-export-only surface. The second is why the allowlist names `__all__`
  rather than permitting assignments generally: a cache and a lock are what
  that file actually held
- `if True:` wrapped around `import threading`, `_cache_lock`, `_providers`
  and `def _get_or_create`, appended to `app/stt/__init__.py` -- **one** test,
  the same surface, reported as `stt/__init__.py:37 If`. It reddened **zero**
  while that gate listed node types to reject instead of statements to accept:
  an `ast.If` is neither a definition nor an assignment, so the block and
  everything nested in it was waved through, and `try: ... except ImportError:`
  and `if TYPE_CHECKING:` carry the same load
- `from app.stt import clear_cache` planted in `app/pipeline/service.py`'s
  `process_audio` -- **one** test, the deferred-import gate, in the absolute
  package spelling
- `from .. import stt` planted in the same function -- **one** test, and it
  reddened **zero** until Stage 5. That spelling already resolves to `app.stt`,
  so the trailing-name strip the bare relative branch applied took it to `app`
  and matched nothing; accepting either resolution is what the branch does now
- `from . import clear_cache` planted in `app/stt/local_setup.py`, the bare
  relative spelling that resolves to a name *inside* the package -- **one**
  test, and the strip is what catches it. The two mutations pull the branch in
  opposite directions, which is why it asks for both
- the hoisted `from app.stt.routing import get_provider, peek_local_provider`
  in `app/stt/local_setup.py` re-deferred into `ensure_local_ready` -- **zero**
  tests here, the negative control for the gate's scope rather than the one
  test this entry claimed until Stage 5. `from app.stt.routing import ...`
  names a module, not the package, so it never reaches the gate. Re-run in
  full: **27 failed, 1403 passed**, none of them in this file -- 25
  `AttributeError: module 'app.stt.local_setup' has no attribute
  'get_provider'` out of `monkeypatch.setattr` in `tests/test_local_setup.py`,
  one more in `tests/test_background_tasks.py`, and a `NameError` in
  `tests/test_pipeline.py` from the two call sites that sit outside the
  function the import moved into. The same statement left at module level
  reddens nothing anywhere

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

_COMPOSITION_ROOT = "app.config"

_MAY_IMPORT_THE_COMPOSITION_ROOT = {"core/config.py"}

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

_RESAMPLING_STACK = frozenset({"soxr", "app.audio.timeline"})

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

_RE_EXPORT_ONLY_PACKAGE_INITS = {"stt": {"__all__"}}

_STT_PACKAGE = "app.stt"

_MAY_DEFER_THE_STT_PACKAGE_IMPORT = {"app/main.py"}

_KNOWN_TWO_NODE_PACKAGE_CYCLES = {
    ("app.preferences", "app.stt"),
}

_MUTUALLY_DEPENDENT_PACKAGES = frozenset(
    {
        frozenset(
            {
                "app.audio",
                "app.config",
                "app.core",
                "app.embeddings",
                "app.preferences",
                "app.stt",
                "app.transcripts",
            }
        ),
    }
)

_NON_FEATURE_PACKAGES = {"core"}

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

    `from app.audio import timeline` names that module as surely as
    `from app.audio.timeline import x` does, and only the second spelling used
    to arrive as `app.audio.timeline` -- the first arrived as `app.audio` and
    walked past every rule written at module granularity. Both the package and
    the submodule are returned now, so the package-level rules keep matching on
    the base while the module-level ones stop being a spelling choice.
    """
    if not node.level:
        bases = [node.module] if node.module else []
    else:
        parts = package.split(".") if package else []
        parts = parts[: max(len(parts) - node.level + 1, 0)]
        if not node.module:
            return [".".join(parts + [alias.name]) for alias in node.names]
        bases = [".".join(parts + node.module.split("."))]

    names = list(bases)
    known = _modules()
    names.extend(
        f"{base}.{alias.name}"
        for base in bases
        for alias in node.names
        if f"{base}.{alias.name}" in known
    )
    return names


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


def _reaches_the_composition_root(imported: str) -> bool:
    """Whether one imported dotted name names the composition root.

    A prefix match, like the downward check's, rather than the equality this
    started as: `app/config.py` is a module today, and the day it becomes
    `app/config/` an equality test stops matching `app.config.runtime` and goes
    silently slack.
    """
    return imported == _COMPOSITION_ROOT or imported.startswith(f"{_COMPOSITION_ROOT}.")


def test_no_core_module_imports_a_feature_package():
    """`app.core` is the layer every package may import, so it must not import
    them back — and since spec 165 nothing here is exempt from that.

    This is the half of the rule spec 165 actually delivered: the exception list
    that granted `config.py` and `router.py` the permission is gone rather than
    narrowed, so there is no allowlist to read alongside the failure. It does
    not make `core` a leaf and is not claimed to — `app.core.config` re-exports
    a settings object assembled from every feature package (ADR 076)."""
    reaching_down = []
    for module, path in _modules().items():
        if not module.startswith("app.core"):
            continue
        for imported in _imported_names(path):
            head = imported.split(".")
            if len(head) >= 2 and head[0] == "app" and head[1] in _FEATURE_PACKAGES:
                reaching_down.append(f"{module} -> {imported}")

    assert not reaching_down, (
        "These app/core modules import a feature package: "
        f"{reaching_down}. There is no exception list any more: the module "
        "belongs outside core (see docs/style-guide.md §1a), or the settings "
        "class it wants belongs in the package that reads it (ADR 073)."
    )


def test_only_one_module_imports_the_composition_root_directly():
    """The singleton has two working spellings and both are policed, with one
    module allowed the upper one.

    Its own function rather than a second assertion inside the rule above,
    because the first `assert` to fire hides the second: a diff that both plants
    a feature-package import in `core` and reaches for `app.config` elsewhere
    would report only the half that ran first.

    Scoped to every module under `app/`, not to `core` alone. `from app.config
    import settings` is plantable in any package, and scoping the walk to `core`
    left it green everywhere else — which is what it did when this gate shipped
    (ADR 076)."""
    reaching_up = []
    for module, path in _modules().items():
        relative = path.relative_to(_APP_DIR).as_posix()
        if relative in _MAY_IMPORT_THE_COMPOSITION_ROOT:
            continue
        for imported in _imported_names(path):
            if _reaches_the_composition_root(imported):
                reaching_up.append(f"{module} -> {imported}")

    assert not reaching_up, (
        f"These modules import {_COMPOSITION_ROOT} directly: {reaching_up}. "
        "Import `app.core.config`, the one doorway, which re-exports the same "
        "objects. Adding a second doorway to _MAY_IMPORT_THE_COMPOSITION_ROOT "
        "is a decision, not a formality — see ADR 076."
    )


def test_the_composition_root_gate_still_matches_something():
    """The mirror the two allowlists below get, for the two constants above.

    The gate matches a dotted string against imports and a path against the
    tree, so renaming or moving either end leaves it matching nothing at all —
    green, with every assertion in it vacuous. That is not hypothetical: before
    this test existed, renaming `app/config.py` to `app/composition.py` and
    repointing the re-export left all fourteen tests in this file passing."""
    stale = []
    if _COMPOSITION_ROOT not in _modules():
        stale.append(f"{_COMPOSITION_ROOT}: no such module")
    for relative in sorted(_MAY_IMPORT_THE_COMPOSITION_ROOT):
        path = _APP_DIR / relative
        if not path.exists():
            stale.append(f"{relative}: no such module")
        elif not any(
            _reaches_the_composition_root(imported)
            for imported in _imported_names(path)
        ):
            stale.append(f"{relative}: does not import {_COMPOSITION_ROOT}")

    assert not stale, (
        f"The composition-root gate describes nothing: {stale}. Repoint "
        "_COMPOSITION_ROOT and _MAY_IMPORT_THE_COMPOSITION_ROOT at where the "
        "singleton is now defined and re-exported — until then the test above "
        "passes without checking anything."
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


def _capture_source_modules() -> list[str]:
    """Every module implementing the system-audio capture contract, found by
    subclass rather than by a hand-kept list.

    `CLAUDE.md` names per-platform loopback capture as work still to come, so a
    list typed out here would exempt the third platform's source by forgetting
    it -- the same failure this file's package walk exists to remove one level
    up.
    """
    found = []
    for path in sorted((_APP_DIR / "audio").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "SystemAudioSource"
                for base in node.bases
            ):
                found.append(path.relative_to(_APP_DIR).as_posix())
                break
    return found


def test_a_capture_source_never_reaches_the_resampling_stack():
    """Both capture callbacks run on the audio thread, and the only thing either
    ever took from `timeline.py` was a deinterleave that touches `soxr` nowhere
    — so importing that module pulled the resampling stack in to do nothing
    with it. The deinterleave lives in `analysis.py` beside the `to_mono` it
    calls, which is the placement fix 084 already chose for `to_mono` itself.

    The static half cannot see a transitive acquisition -- `soxr` appearing in
    any of the four sibling modules these two import would put the stack back
    with no offender named -- so the probe below asks the process instead.
    Only `app.audio.macos_tap` can answer it on every runner: importing
    `windows_loopback` needs the Windows-only `pyaudiowpatch` wheel.
    """
    modules = _capture_source_modules()
    assert modules, (
        "no SystemAudioSource implementation was found under app/audio/ -- the "
        "walk is broken, which would make this test pass by finding nothing."
    )

    offenders = []
    for relative in modules:
        for imported in _imported_names(_APP_DIR / relative):
            if imported in _RESAMPLING_STACK:
                offenders.append(f"{relative} imports {imported}")

    assert not offenders, (
        f"{offenders}. A capture callback has no resampling to do; shared code "
        "it needs moves down into `app.audio.analysis` instead."
    )

    assert_import_loads_no_module(
        "app.audio.macos_tap", ("soxr", "app.audio.timeline")
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


def _module_level_assigned_names(node: ast.stmt) -> list[str]:
    """Every name one statement binds by assignment, in any of the three
    spellings -- `x = ...`, `x: T = ...` and `x += ...`. A tuple target binds
    each of its names, so the walk is over the target rather than over a single
    `Name`."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return []
    return sorted(
        {
            name.id
            for target in targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
    )


def test_a_re_export_only_package_init_declares_names_rather_than_defining_them():
    """`docs/style-guide.md` §1a: a package `__init__.py` re-exports; it does
    not implement. `app/stt/__init__.py` is deliberately absent from
    `_IMPORT_FREE_PACKAGE_INITS` above because re-exporting is what it is for
    -- which left the weaker half of the rule with nothing enforcing it, and
    the file grew a routing layer, a provider cache and a `threading.Lock`
    inside it. `_RE_EXPORT_ONLY_PACKAGE_INITS` names `stt` alone:
    `app/embeddings/__init__.py` still holds that same shape, so listing it
    would fail this gate rather than pin it.

    A re-export surface is a list of names: imports, an `__all__`, a docstring.
    Everything else is an implementation, and the module it belongs in is a
    sibling with a name -- `app/stt/routing.py`. The allowlist is per package
    and names the assignments each may keep, so publishing `__all__` stays
    legal and planting a cache beside it does not.

    An allowlist over statements, not a list of node types to reject. Rejecting
    `FunctionDef`, `ClassDef` and a bare assignment accepted everything it did
    not recognise: an `if True:` block holding `_cache_lock`, `_providers` and
    `_get_or_create` is an `ast.If`, which is none of the three, and the same
    goes for the `try: … except ImportError:` and `if TYPE_CHECKING:` blocks
    that are the likeliest accidental carriers. Classifying the top-level body
    is exhaustive under the inverted form, because no accepted statement can
    contain another: an import and an `__all__` assignment have no body, so a
    nested definition is only reachable through a compound statement that is
    itself the offender. This is the shape
    `tests/test_settings_isolation.py`'s
    `test_the_core_config_module_holds_nothing_but_its_re_export` already
    implements, and the docstring-strip idiom below was taken from it."""
    offenders = []
    for package, allowed in sorted(_RE_EXPORT_ONLY_PACKAGE_INITS.items()):
        path = _APP_DIR / package / "__init__.py"
        assert path.exists(), (
            f"{package}/__init__.py no longer exists — deleting it turns "
            f"{package} into a namespace package, which drops the re-export "
            "surface this pins rather than satisfying it. Update this test."
        )
        body = list(ast.parse(path.read_text(encoding="utf-8")).body)
        if body and ast.get_docstring(ast.Module(body=body, type_ignores=[])):
            body = body[1:]
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            assigned = _module_level_assigned_names(node)
            if assigned and all(name in allowed for name in assigned):
                continue
            described = (
                f"assigns {', '.join(assigned)}" if assigned else _statement_description(node)
            )
            offenders.append(f"{package}/__init__.py:{node.lineno} {described}")

    assert not offenders, (
        f"These re-export surfaces implement something: {offenders}. A package "
        "`__init__.py` listed in _RE_EXPORT_ONLY_PACKAGE_INITS holds a "
        "docstring, imports and the names it publishes — nothing else, in any "
        "position, including inside an `if` or a `try` block. The behaviour "
        "belongs in a sibling module, which is what `app/stt/routing.py` is. "
        "See docs/style-guide.md §1a."
    )


def _function_body_imports(path: Path) -> list[ast.stmt]:
    """Every import statement written inside a function body, deduplicated.

    A function nested in another function is reached twice by the outer walk,
    so the statements are keyed by identity rather than appended blindly; a
    duplicate would report the same site twice and make the message read as two
    defects."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[int, ast.stmt] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Import, ast.ImportFrom)):
                found[id(inner)] = inner
    return sorted(found.values(), key=lambda statement: statement.lineno)


def _names_the_stt_package(node: ast.stmt, package: str) -> bool:
    """Whether one import statement names `app.stt` itself.

    `from app.stt import clear_cache` and `from app.stt import
    local_whisper_cpp_cmd` both do, in the second case alongside the submodule;
    `from app.stt.local_factory import ...` does not, and neither does `import
    app.stt.local_setup`, which names a module that happens to live in the
    package.

    The bare relative spelling has two resolutions and both name the package,
    so the branch accepts either. `from . import clear_cache` inside
    `app/stt/` resolves to the imported *name* (`app.stt.clear_cache`), and
    stripping the trailing segment is what turns it into the package;
    `from .. import stt` written outside the package already resolves to
    `app.stt` itself, and stripping it there yields `app` and matches nothing,
    which is how an unconditional strip lost that spelling. Both stay confined
    to the bare form: `from app.stt.routing import get_provider` strips to
    `app.stt` too, and this gate is scoped to the package rather than to every
    function-local `from app.…`."""
    if isinstance(node, ast.ImportFrom):
        names = _import_from_names(node, package)
        if _STT_PACKAGE in names:
            return True
        return (
            bool(node.level)
            and not node.module
            and any(name.rsplit(".", 1)[0] == _STT_PACKAGE for name in names)
        )
    return any(alias.name == _STT_PACKAGE for alias in node.names)


def test_only_the_sidecar_entry_point_defers_an_import_of_the_stt_package():
    """`docs/style-guide.md`: a function-local `from app.…` is a cycle being
    hidden and should be read as a defect. Seven sites in three files deferred
    an `app.stt` import while the package held the routing layer; six of them
    had nowhere else to go, because the names they wanted existed only in
    `__init__.py`.

    Scoped to the `app.stt` package, not to the general rule it quotes. A
    deferral naming a *module* — `from app.stt.routing import get_provider`,
    `from app.core.gpu_probe import GpuVendor` — walks past this gate and is
    meant to: catching every function-local `from app.…` needs a measured
    allowlist keyed on file and imported module, which eleven such deferrals in
    this package alone would have to be read into first.

    `main.py`'s entry is not a startup-time exemption, whatever the style guide
    says. Measured: `import app.main` leaves `app.stt`, `app.stt.routing` and
    `app.stt.local_setup` in `sys.modules` through the module-level
    `from app.stt.router import router` at `app/main.py:31`, so by the time
    `lifespan()` runs there is nothing left to defer. What the deferral buys is
    **late binding** — `monkeypatch.setattr(app.stt, "clear_cache", …)` at
    `tests/test_background_tasks.py:392` and `:476` and
    `tests/test_startup.py:262` replaces the attribute on the module object,
    and only a name read inside the shutdown body sees the replacement.

    The set of files is pinned, not a count: a second deferral in `main.py`
    would be the same decision already taken, while the first one anywhere else
    is the defect. The walk is the one the cycle tests use, so a spelling that
    hides from this gate hides from those too."""
    measured = {}
    for path in sorted(_modules().values()):
        relative = path.relative_to(_APP_DIR.parent).as_posix()
        package = _containing_package(path)
        sites = [
            f"{relative}:{node.lineno} {_statement_description(node)}"
            for node in _function_body_imports(path)
            if _names_the_stt_package(node, package)
        ]
        if sites:
            measured[relative] = sites

    assert set(measured) == _MAY_DEFER_THE_STT_PACKAGE_IMPORT, (
        f"pinned:   {sorted(_MAY_DEFER_THE_STT_PACKAGE_IMPORT)}; "
        f"measured: {sorted(measured)}. Sites: "
        f"{sorted(site for sites in measured.values() for site in sites)}. "
        "Import `app.stt` — or the module that actually holds the name, which "
        "for the routing layer is `app.stt.routing` — at module level. Adding "
        "a file to _MAY_DEFER_THE_STT_PACKAGE_IMPORT claims the deferral buys "
        "late binding the way `main.py`'s does, where the shutdown body reads "
        "`clear_cache` through the module object three tests replace it on; it "
        "does not buy startup time, because `app/main.py:31` has already "
        "imported the package by then. A file leaving the measured set means "
        "its entry is now dead and should go."
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


def _two_node_package_cycles() -> set[tuple[str, str]]:
    """Every pair of packages that import each other directly.

    Two nodes and no more, which is the whole field of view of the two tests
    below: a loop running `app.core -> app.config -> app.audio -> app.core` is
    a cycle this returns nothing for. Nineteen of the twenty elementary cycles
    in this graph are of that kind. `app.main` is excluded because building the
    application means importing every router.
    """
    edges = _package_edges()
    return {
        tuple(sorted(pair))
        for pair in edges
        if (pair[1], pair[0]) in edges and "app.main" not in pair
    }


def _mutually_dependent_packages() -> frozenset[frozenset[str]]:
    """Every set of packages each of which reaches every other.

    The transitive closure of the package graph, grouped by mutual
    reachability. Computed rather than enumerated: a list of elementary cycles
    holds 20 entries here and 44 after one added function-local import, so it
    records a snapshot rather than a rule, while this membership moves by one
    readable name on that same edge.

    `app.main` is excluded, as `_two_node_package_cycles` excludes it.
    """
    edges = _package_edges()
    nodes = {package for pair in edges for package in pair} - {"app.main"}
    reaches = {
        node: {target for source, target in edges if source == node and target in nodes}
        for node in nodes
    }
    growing = True
    while growing:
        growing = False
        for node in nodes:
            grown = set(reaches[node]).union(*(reaches[t] for t in reaches[node]))
            if grown != reaches[node]:
                reaches[node] = grown
                growing = True

    components = {
        frozenset(
            other for other in nodes if other in reaches[node] and node in reaches[other]
        )
        for node in nodes
    }
    return frozenset(component for component in components if len(component) > 1)


def test_no_two_node_package_cycle_beyond_the_ones_already_accounted_for():
    """A new pair here means a module was placed where it makes two packages
    depend on each other directly, which is the defect that made `core`
    unreadable in the first place.

    This counts pairs, and nothing else. It is not a measure of how tangled the
    package graph is — the test below is — and its falling count has been read
    as one: spec 165 took this list from four entries to one while leaving the
    elementary-cycle count at 20, because the three loops it removed came back
    one hop longer through the composition root."""
    new = _two_node_package_cycles() - {
        tuple(sorted(pair)) for pair in _KNOWN_TWO_NODE_PACKAGE_CYCLES
    }
    assert not new, (
        f"New two-node package cycles: {sorted(new)}. Every remaining pair is "
        "listed in _KNOWN_TWO_NODE_PACKAGE_CYCLES with the reason it survives; "
        "adding to that list is a decision, not a formality."
    )


def test_the_known_two_node_cycle_list_does_not_outlive_the_cycles():
    """The other direction: a cycle that has been fixed must leave this list,
    or the list stops describing anything and the test above goes slack."""
    stale = {
        tuple(sorted(pair)) for pair in _KNOWN_TWO_NODE_PACKAGE_CYCLES
    } - _two_node_package_cycles()

    assert not stale, (
        f"These cycles no longer exist and should be removed from "
        f"_KNOWN_TWO_NODE_PACKAGE_CYCLES: {sorted(stale)}"
    )


def test_every_backend_package_is_classified_as_feature_or_not():
    """The feature-package rule above only sees the names in
    `_FEATURE_PACKAGES`, so a package absent from it is exempt in full rather
    than checked.

    The same hole `test_every_backend_package_is_covered_by_the_web_framework_allowlist`
    closes for the web-framework gate, and it is not hypothetical here:
    `app/llm/` existed until JS-169 deleted it, so packages do get added and
    removed. Without this, a new `app/newpkg/` could be imported from inside
    `core` and every rule in this file would stay green.

    Mutation-checked: creating a directory under `app/` that is named in
    neither set reddens this test and nothing else.
    """
    unclassified = [
        package
        for package in _package_directories()
        if package not in _FEATURE_PACKAGES and package not in _NON_FEATURE_PACKAGES
    ]

    assert not unclassified, (
        f"These packages are in neither set, so the feature-package rule does "
        f"not see them: {unclassified}. Add each one to _FEATURE_PACKAGES, or "
        "to _NON_FEATURE_PACKAGES if it is shared ground every package may "
        "import."
    )


def test_the_mutually_dependent_package_set_has_not_changed():
    """Seven packages all reach each other, so none can be read, moved or
    tested without the other six. This pins which seven.

    Every such group is pinned, not the largest one. Returning only the biggest
    left a second tangle invisible and made the answer depend on dictionary
    order when two tied on size, so a three-package loop sharing no edge with
    the seven passed unseen and the test could change verdict between runs on
    interpreter start-up alone.

    The instrument the two-node tests are not. Compared as an exact set, so it
    fails in both directions: a package joining a group means a change made the
    tangle bigger and has to say so, and a package leaving means a spec
    genuinely untangled something and this constant is the stale half.

    Had this test existed before spec 165 it would have reddened on that branch
    — six names pinned, seven measured — because moving the composition root to
    `app/config.py` put a new package inside the component while the two-node
    count fell from four to one. That disclosure is the reason it exists
    (ADR 076)."""
    measured = _mutually_dependent_packages()

    assert measured == _MUTUALLY_DEPENDENT_PACKAGES, (
        f"pinned:   {sorted(sorted(group) for group in _MUTUALLY_DEPENDENT_PACKAGES)}; "
        f"measured: {sorted(sorted(group) for group in measured)}. "
        "Update _MUTUALLY_DEPENDENT_PACKAGES deliberately and say in the spec "
        "which direction it moved: growing it is a cost this change is paying, "
        "shrinking it is progress worth naming. See ADR 076."
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
