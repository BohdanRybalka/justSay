"""The structural rules spec 076 established, enforced instead of documented.

`docs/style-guide.md` §1a states where a backend module goes, and ADR 044
records why. Prose rots; this file fails.

Fifteen properties are pinned here:

1. No module of shared ground imports a feature package or the HTTP boundary,
   with no exception at all: shared ground is the layer every other package may
   import, and importing one back is how `core` came to hold the transcript
   store, the user preferences and four routers at once. The walk is stated
   over `_NON_FEATURE_PACKAGES` rather than over the name `core`, so a package
   refiled as shared ground is measured against the rule instead of leaving it.
   `app.api` is classified as neither a feature package nor shared ground,
   because shared ground is what `core` may import and a router is not.
   Separately, only the modules `_MAY_IMPORT_THE_COMPOSITION_ROOT` names
   import the composition root `app.config` directly — `main.py`,
   `api/router.py` and `api/auth_middleware.py`, application assembly and the
   HTTP boundary, which are the readers of the application-level fields no
   feature package owns. That half is scoped to all of `app/` rather than to
   `core`, because the spelling is plantable in any package, and it
   prefix-matches so `app.config.runtime` could not walk past if
   `app/config.py` ever became `app/config/`. A feature package reads the
   settings instance its own package holds instead of the aggregate, which is
   what leaves `core` importing no feature package at all (ADR 091).
2. `app.audio.analysis` imports nothing the frozen PyInstaller sidecar lacks,
   and is imported *from* rather than importing — its own docstring names the
   libraries that would break the packaged build, and ADR 015 depends on it.
   A violation here ships broken; only a tag push would otherwise reveal it.
3. No package acquires a web framework, `fastapi` and `starlette` alike,
   outside the modules each package's exempt set names. Every directory under
   `app/` holding any `.py` file is a key in that allowlist — an `__init__.py`
   is not required, so a PEP 420 namespace package cannot be exempt by being
   forgotten — and no exemption survives the import it covers, nor the package
   it names. Shared ground carries no exempt set at all, pinned by its own
   test: the allowlist accepts whatever a future diff writes into it, so the
   emptiness is asserted over the classification rather than over one package
   name, and a key missing altogether is the same defect as a key with an entry
   in it (ADR 082). The modules sitting
   directly under `app/` are checked at the same time, with `main.py` the one
   exemption: building the FastAPI application is its job. The composition root
   is `app/config.py`, a different module.
4. The set of *two-node* package cycles does not grow and does not outlive the
   pairs it lists, and — separately — the set of packages that all reach each
   other does not change in either direction. The first instrument sees one
   pair today; three of this graph's four elementary cycles run longer
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
8. A package `__init__.py` named as re-export-only holds a docstring, the
   `__all__` it publishes, and imports that bind exactly those names out of
   its own modules — nothing else in any position. It is an allowlist rather
   than a list of banned node types: a cache and a lock wrapped in
   `if TYPE_CHECKING:` or in `try: … except ImportError:` are neither a
   definition nor a top-level assignment, and a gate that classifies what it
   recognises accepts everything it does not. The import arm is an allowlist
   for the same reason and shipped without being one: skipping every import on
   node type let `from app.stt.local_whisper_cpp import
   WhisperCppServerSTTProvider` charge the whisper.cpp provider stack to every
   `import app.stt.<anything>`. Its other half is property 12. `app/stt/__init__.py`
   held a routing layer, a provider cache and a `threading.Lock` across 215
   lines, which is why the HTTP router could not take the obvious name and why
   seven call sites imported the package from inside a function body. This is
   the weaker sibling of property 5: the packages listed there hold nothing at
   all, the ones listed here hold a list of names and no behaviour. The
   allowlist covers `stt` alone; `app/embeddings/__init__.py` still holds the
   same shape — two locks, three cache globals, a Protocol and two functions —
   and adding it here would fail rather than pin, because making it pass means
   splitting that package too.
9. Every function-body import of the `app.stt` package *or of a module inside
   it* is one this file records by name. The package alone stopped being
   enough the moment the routing layer left `__init__.py`: the eight names it
   pins live in `app.stt.routing` now, so re-deferring `from app.stt.routing
   import get_provider, peek_local_provider` walked past a gate written for
   `from app.stt import …` while being the same defect one dot further along.
   The allowlist is keyed on the names each file may defer rather than on the
   file, so a file already holding a recorded deferral cannot acquire a
   second, different one for free. Seven files hold one: four reach the
   prewarm entry points in `app.stt.local_setup`, three reach the platform
   factory and the provider classes five test sites replace by string path,
   and `app/main.py` reaches `app.stt.routing`, where the deferral buys late
   binding rather than startup time -- `import app.main` has already loaded
   the package through `app/main.py:31` before `lifespan()` runs, and three
   tests replace `clear_cache` on the module object the shutdown body reads it
   from. `docs/style-guide.md` reads every other function-local `from app.…`
   as a cycle being hidden; this gate does not enforce that general rule, only
   the `app.stt` half of it, because that is the package that had seven of
   them.
10. A system-audio capture source reaches neither `soxr` nor the module that
   imports it. Both capture callbacks run on the audio thread and neither has
   any resampling to do; the deinterleave they share lives in `analysis.py`
   beside the `to_mono` it calls. The sources are found by subclass rather
   than by a typed list, so the next platform's is covered the day it is
   written.
11. No module under `app/` takes a routing name off the `app.stt` package.
   The re-export in `app/stt/__init__.py` gives the eight routing names a
   second live address, and a second address is a second `monkeypatch`
   target: a test replacing `app.stt.clear_cache` to assert "the cache was
   cleared on mode switch" passes while asserting nothing, if the module
   under test bound the same function through `app.stt.routing`. Every module
   under `app/` takes those names from `app.stt.routing`, with no exception:
   `app/main.py` was one until its shutdown body took the same late binding
   from the routing module instead, which costs nothing and leaves the
   allowlist empty. What that pins is one import *spelling*, and it was
   called one address for two review rounds while never being one -- `from
   app.stt.routing import clear_cache` keeps a copy of its own, and counting
   those copies is property 13. `from app.stt import *` takes every one of
   them and reported nothing, because matching each alias against the
   re-export set reads `*` as a name nobody publishes. The names are read off
   the re-export block rather than listed here, so one added there is covered
   the day it is written -- and *which* names that block yields is itself a
   selection that emptied this gate twice without emptying the set: a
   `from .routing import <name>` resolved by string comparison rather than as
   an import, and a name published out of a third module inside the package.
   The block is resolved through the same helper every other rule here uses
   now, and what it yields is checked against what `app/stt/routing.py`
   defines.
12. `app/stt/__init__.py` publishes exactly the names it imports, as the same
   objects. `__all__` and the import block above it are two halves of one
   surface and nothing read them against each other, so a name dropped from
   the import block and left in `__all__` made `from app.stt import *` raise
   `AttributeError` at runtime with every gate here green. Property 8 closes
   the half where an import runs ahead of `__all__`; this closes the half
   where `__all__` promises what no import binds, and the identity of what it
   does bind. That identity was asserted to hold structurally and did not: an
   aliased re-export -- `from app.stt.routing import get_routed_provider as
   get_provider` -- publishes a promised name bound to a different function
   and satisfies every name-level check, and a second `__all__ = __all__ +
   ["Bogus"]` widens the surface through a value no walk over list elements
   reads. This property alone is measured off the imported package rather
   than off its tree, and that is why no spelling dodges it: three review
   rounds each found a different one walking past a walk written for the
   round before.
13. A routing name has no third address under `app/`. Property 11 is about
   where a module imports a name from; this is about how many modules keep a
   copy of it, and the two are not the same measurement. Hoisting five
   deferred imports in `app/stt/local_setup.py` onto the spelling property 11
   endorses still left four routing functions at `app.stt.local_setup.<name>`,
   forty-six `monkeypatch` targets moved onto that copy, and a patch at
   `app.stt.routing.is_model_loaded` stopped being seen by `check_status()`.
   Binding the module instead -- `from app.stt import routing`, then
   `routing.<name>()` -- reads the one attribute at call time and makes no
   copy. Two modules keep one and both predate this pin, checkably: on
   `2e2d099` each bound the same names off the package.
   What it counts is a module attribute bound by an import or by a plain
   module-level assignment, from any source module at all. It filtered on the
   source for one round -- skipping any statement whose names led nowhere
   inside `app.stt` -- and `from app.pipeline.service import
   get_routed_provider` planted a fourth live `monkeypatch` target that a gate
   named for counting addresses reported nothing for. It counts names, not
   objects, so a same-named function from another package is reported too;
   that over-count is an allowlist entry a reviewer reads, while an
   under-count is the defect. An address built by an expression -- a class
   attribute, a dict value, a default argument, `setattr` on the module
   object, a name computed at run time -- is past a static walk and is not
   claimed. The two shapes that would hide a whole namespace rather than one
   name, a module-level star import and a module-level `__getattr__`, are
   failures rather than misses.
14. The HTTP boundary is imported by the module that builds the application and
   by nothing else, imports no feature package, and holds nothing but boundary
   code. The first two are property 1 read from the other two sides: shared
   ground not importing `app.api` leaves a feature package importing it, and a
   router importing a feature package, both unchecked, and either one re-creates
   the inversion the package was cut out of `core` to remove. The third is
   stated over what reaches a module rather than over its exempt-set entry: a
   module here is imported by the application builder or by another module of
   the boundary, and one reached by neither is not routes, middleware or an
   exception handler whatever it holds. Boundary code is free to import no web
   framework — a pydantic response model is the shape that arrives — and an
   inventory over the exempt set left such a module no configuration this file
   accepts at all, since the mirror in property 3 fails an entry covering no
   import. Each of the three walks pins itself non-empty, and the allowlist
   naming who may import the boundary is mirrored against the tree the way the
   composition root's is: an entry written ahead of the import it covers, or
   left behind after that import moved, is the defect emptying the set is not
   (ADR 079).
15. The three classification sets are pairwise disjoint, shared ground is
   non-empty and every name in it is a real directory under `app/`, and the
   boundary's own two constants are mirrored against the tree in the same way.
   Which set a package sits in is the single input every rule above reads, so
   a name moved between them, or written for a directory that does not exist,
   changes what is checked while leaving the rules themselves untouched.
   Shared ground is not pinned by name: `core` is the only member today, and
   asserting that spelling would have to be re-edited by the same diff that
   adds a second one, which is the edit the rule exists to catch.

Every assertion below was mutation-checked when written. The list below is a
ledger of mutations that were actually run, against the module actually named,
with the number of tests each one reddens:

- a core module made to import a feature package, in the absolute
  (`from app.audio import analysis`) and the relative (`from ..audio import
  analysis`) spelling alike -- **three** tests each, measured in
  `app/core/utils.py`: the feature-package rule, the two-node cycle test and
  the component pin, because `app.core` sits outside the component now and an
  edge into `app.audio` builds a fresh one out of the two of them
- `from app.audio.config import AudioSettings` planted in `app/core/utils.py`
  -- **three** tests: the feature-package rule, the two-node cycle test below,
  because `app.core <-> app.audio` left `_KNOWN_TWO_NODE_PACKAGE_CYCLES` in the
  same change and so is a new pair again, and the component pin, which reports
  that pair as a group of its own. Both halves of the `99e05e4` counter-claim
  were re-measured against a
  `git archive` of that commit. Planted in the module its exemption covered,
  the one this gate has since dropped, the same line reddened **zero** there.
  Planted in `utils.py`, which that exemption never covered, it reddened
  **one**
- `from app.config import settings` planted in `app/transcripts/search.py`
  beside the slice import it already holds, so the module still works and only
  the spelling reaches the aggregate -- **two** tests, the composition-root
  rule and the component pin, the second because that edge pulls `app.config`
  and everything it imports back inside the group. It is planted outside
  `core` on purpose: the gate walked `app.core` alone when it shipped, so this
  exact line was green in seven packages out of eight
- `from app.config.runtime import settings` planted in `app/core/utils.py`'s
  `sse_event` body -- **two** tests, the composition-root rule and the
  component pin. The submodule does not exist and is not meant to; what it
  pins is that the upward check prefix-matches rather than comparing for
  equality, which is how it shipped
- both of the above in one diff -- `from app.audio.config import AudioSettings`
  and `from app.config import settings` added to `app/core/utils.py` together
  -- **four** tests, and the point of the mutation is that the feature-package
  rule and the composition-root rule report *both* messages. They were one test
  function with two asserts until Stage 5, where the first `assert` firing hid
  the second
- `app/config.py` renamed to `app/composition.py` with its three importers
  repointed -- **no** test of this file reports, because the file no longer
  collects: `tests/conftest.py` imports the composition root by name and the
  run stops at `ModuleNotFoundError: No module named 'app.config'`. The mirror
  is what catches a rename that does collect; before it existed the same rename
  left all fourteen tests here green, with the gate matching nothing at all
- a function-local `from app.pipeline import service` added to
  `app/stt/base.py` -- **two** tests, the component pin (four names to six,
  `app.audio` arriving alongside `app.pipeline`) and the two-node list, since
  `app.pipeline <-> app.stt` is also a new pair. The same edge takes the
  elementary-cycle enumeration from 4 to 11, which is why membership is pinned
  and the enumeration is not
- `import fastapi` planted in `app/audio/analysis.py`, the base DSP module --
  three tests, because that module is a non-exempt file of a
  web-framework-free package, is the module property 2 guards, and sits on
  `app.audio.timeline`'s import path
- `from starlette.requests import Request` planted in the same module -- two
  tests, the same first two
- `app/audio/analysis.py` made to import `app.audio.timeline`, absolutely and
  relatively (`from .timeline import ...`) -- one test each
- a fresh `transcripts <-> pipeline` cycle -- **two** tests, the two-node list
  and the component pin, which reports `app.pipeline` joining the group; a
  fictional entry added to the known-cycle list -- **one**, the staleness half
- a provider given `HTTPException` -- one test
- a `fastapi` importer added as `app/handlers.py`, directly under `app/`, and
  as `app/newpkg/thing.py` in a directory with no `__init__.py` -- one test
  each
- a fictional file added to a feature package's exempt set, and a fictional
  package key carrying an empty exempt set -- one test each. Deleting a real
  package key is not always one test: the `core` key deleted reddens **two**,
  the coverage gate and the emptiness pin, which reads the dict through `.get`
  so a key missing altogether reports that pin's message instead of a
  `KeyError`. The `api` key deleted reddens **one**, the coverage gate alone,
  which is the gate whose message names the missing key rather than the
  modules it covered
- `import fastapi` planted in `app/core/utils.py` -- **one** test, the
  web-framework gate. `core` names no exempt module any more, so there is
  nothing for the plant to hide behind
- `"utils.py"` written back into `_WEB_FRAMEWORK_FREE_PACKAGES["core"]` with
  that plant left in place -- **one** test, the emptiness pin alone. The
  exemption then covers an import that really exists, so the coverage gate and
  the mirror above are both satisfied and nothing else in this file reports.
  The same entry written back *without* the plant -- **two**, the emptiness pin
  and the mirror
- `from app.api.router import router` planted in `app/core/utils.py` --
  **four** tests: the rule over shared ground, the rule over who may import the
  boundary, the two-node list, and the component pin. Re-run with `api` moved
  out of `_HTTP_BOUNDARY_PACKAGES` and into `_NON_FEATURE_PACKAGES`, the rule
  over shared ground does still go silent, and **seven** report in its place --
  the two cycle instruments, the boundary's two walks finding nothing left to
  check, the rule over who may import the boundary, the boundary's own mirror,
  and the emptiness pin, which now reads `api` as shared ground carrying three
  exemptions. The silence of that one rule is no longer the whole of the
  mutation
- `api` moved into `_NON_FEATURE_PACKAGES` with no import planted -- **five**
  tests, those seven less the two cycle instruments; `api` added there while
  left in `_HTTP_BOUNDARY_PACKAGES` -- **two**, the overlap half of the
  classification pin and the emptiness pin; `audio` moved from
  `_FEATURE_PACKAGES` into `_NON_FEATURE_PACKAGES` -- **two**, the rule over
  shared ground and the emptiness pin; `embeddings` moved the same way --
  **one**, the rule over shared ground alone, because that package carries no
  exemption for the emptiness pin to see. While that rule was spelled
  `app.core` by name the `embeddings` move reddened **zero** and the `audio`
  move reddened **one**, the emptiness pin on its own
- emptying `_HTTP_BOUNDARY_PACKAGES`, which is also what leaving `api` out of
  all three classification sets amounts to while that set names it alone --
  **five** tests: the classification gate, the boundary's two walks, the rule
  over who may import the boundary, and the boundary's mirror. The allowlist
  coverage gate is **not** among them: `api` is still a key in
  `_WEB_FRAMEWORK_FREE_PACKAGES` and that gate reports only packages absent
  from the dict. An import and an `__all__` added to `app/api/__init__.py` --
  one test
- `from app.api.error_handler import register_error_handlers` planted at the
  top of `app/pipeline/service.py` -- **one** test, the rule over who may import
  the boundary, and **zero** before that rule existed
- `_MAY_IMPORT_THE_HTTP_BOUNDARY` emptied, so `app/main.py` itself offends --
  **three** tests: that rule, the reachability rule, and the allowlist's own
  mirror reporting that it walked nothing. `core/utils.py` and
  `pipeline/nonexistent.py` added to it beside `main.py` -- one real module
  that does not import the boundary and one path that does not exist --
  **one** test, that same mirror, and **zero** before the mirror existed
- `from app.audio.config import AudioSettings` planted in `app/api/router.py`
  -- **one** test, the boundary's downward rule, and **zero** before it existed
- `app/api/helper.py` added, holding a function, importing no web framework and
  imported by nothing -- **one** test, the reachability rule. Named in `api`'s
  exempt set -- **two**, that rule again and the exemption mirror, because the
  entry covers no import. `app/api/schemas.py` added instead, holding a
  pydantic response model: **one** while nothing imports it, and **zero** once
  `app/api/router.py` does. That last count is the configuration the
  exempt-set inventory this rule replaced left no room for. A pair of modules
  added to `app/api/` importing only each other -- **one** test, the same rule,
  and **zero** while reachability was read one hop deep instead of followed
  from `app/main.py` through the package
- `_NON_FEATURE_PACKAGES` emptied -- **four** tests: the rule over shared
  ground, the emptiness pin, the classification gate and the shared-ground pin.
  `core` replaced there by a fictional `shared` carrying an empty exempt set --
  **four**: the same rule over shared ground finding nothing, the exemption
  mirror, the classification gate and the shared-ground pin. `core` moved out
  of shared ground and into `_FEATURE_PACKAGES` -- **six**
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
- `__all__[0] = threading.Lock()` appended to `app/stt/__init__.py` -- **one**
  test, the re-export-only surface, and **zero** while the allowlist walked
  the whole target for an `ast.Name`. A subscript target binds nothing and
  runs a call instead, yet arrived at the allowlist as the string `__all__`
  and was read as the statement that publishes the surface
- `__all__ += ["STTProvider"]` appended to the same file -- **one** test, and
  **zero** before, for the same reason in the other spelling: an augmented
  assignment mutates in place, and a re-export surface is published once
- `from app.stt import clear_cache` planted in `app/pipeline/service.py`'s
  `process_audio` -- **two** tests, the deferred-import gate in the absolute
  package spelling and the single-address gate, which reads module level and
  function bodies alike
- `from .. import stt` planted in the same function -- **two** tests, the same
  pair. It reddened **zero** until Stage 5, when the bare relative branch
  stripped the trailing name unconditionally and took a spelling that already
  resolves to `app.stt` down to `app`; there is no strip left to get wrong,
  because the match is now the package or anything under it
- `from . import clear_cache` planted in `app/stt/local_setup.py`, the bare
  relative spelling that resolves to a name *inside* the package -- **two**
  tests, and prefix-matching is what catches it in both. Under the branch it
  replaced this one needed the strip and the entry above needed the strip
  skipped, which is the shape of a rule that had to know which way to guess
- the hoisted `from app.stt.routing import get_provider, peek_local_provider`
  in `app/stt/local_setup.py` re-deferred into `ensure_local_ready` -- **one**
  test, and **zero** through two review passes. `from app.stt.routing import
  ...` names a module rather than the package, so a gate matching `app.stt`
  exactly never saw the eight names it exists to keep at module level. What
  caught it instead was 27 incidental failures elsewhere (**27 failed, 1403
  passed**: 25 `AttributeError: module 'app.stt.local_setup' has no attribute
  'get_provider'` out of `monkeypatch.setattr` in `tests/test_local_setup.py`,
  one in `tests/test_background_tasks.py`, and a `NameError` in
  `tests/test_pipeline.py` from the two call sites outside the function the
  import moved into) -- every one of which disappears the day those targets
  are re-pointed. The same statement left at module level reddens nothing
- `app/stt/router.py`'s `from app.stt.routing import clear_cache, get_provider`
  written back as `from app.stt import ...`, and the same for
  `app/pipeline/service.py`'s `get_routed_provider, is_local_provider` -- one
  test each, the import-source gate. Both spellings were the shipped ones
  until this pass and no gate had an opinion
- `import app.stt.local_setup` planted in `app/pipeline/service.py` with
  `app.stt.clear_cache()` beside it -- **one** test, the import-source gate,
  and **zero** while its `ast.Import` arm compared against the string
  `app.stt` for equality. The longer spelling binds `app` exactly as the
  shorter one does, so every routing name was reachable as
  `app.stt.clear_cache` with this file at 22 passed
- `__all__ = __all__ + ["Bogus"]` appended to `app/stt/__init__.py` -- **two**
  tests, the re-export surface and the published-names check, and **zero**
  before: the surface reader returned on the first `__all__`, and the second
  one's value is a `BinOp` that a walk over list elements reads as publishing
  nothing. `from app.stt import *` answered `AttributeError: module 'app.stt'
  has no attribute 'Bogus'` at 22 passed
- `from app.stt.routing import get_routed_provider as get_provider` appended to
  the same file -- **one** test, the published-names check, and **zero**
  before. Every name-level rule here reads the *bound* name, which the alias
  satisfies; `app.stt.get_provider.__name__` was `get_routed_provider` and
  `app.stt.get_provider is app.stt.routing.get_provider` was False at 22
  passed. It is the mutation that made this one property read the imported
  package rather than its tree
- `app/stt/local_setup.py`'s `from app.stt import routing` written back as the
  module-level `from app.stt.routing import get_local_load_error,
  get_provider, is_model_loaded, peek_local_provider` it shipped as, with the
  call sites unqualified again -- **one** test here, the third-address gate,
  and **21** in `tests/test_local_setup.py`, whose `monkeypatch` targets then
  name an address `check_status()` no longer reads. The import-source gate
  stays green through it, which is the whole reason the third-address gate is
  a separate property
- a fictional `app.stt.zz_fictional` added to `_MAY_DEFER_AN_STT_IMPORT`'s
  two-name `app/main.py` entry -- **one** test, and the message names that
  entry alone. It used to print all three recorded names under "entries with
  nothing left to cover", reporting two live deferrals as dead

Six mutations below are the fifth review round, and every one of them is a
hole in *name selection* rather than in what the gates then do with the names.
All six ran green at **23 passed** against the round-4 file:

- `peek_local_provider` moved to `from .routing import peek_local_provider` in
  `app/stt/__init__.py`, with `app/stt/local_setup.py` taking it off the
  package beside `routing` -- **two** tests, the import-source gate and the
  third-address gate, and **zero** before. The re-export set compared
  `node.module` against the dotted string, so the relative spelling took that
  name out of the set *both* gates measure against while leaving the set
  non-empty, which is why neither `assert re_exported` tripwire fired
- `from app.pipeline.service import get_routed_provider` with `_probe =
  get_routed_provider` appended to `app/pipeline/router.py` -- **one** test,
  the third-address gate, and **zero** before. It skipped any statement whose
  resolved names led nowhere inside `app.stt`, so a copy taken from the module
  that already holds one was invisible; `app.pipeline.router
  .get_routed_provider is app.stt.routing.get_routed_provider` was True at 23
  passed, a fourth live target for a gate that exists to count them
- `from app.stt import *` appended to `app/pipeline/router.py` -- **two**
  tests, the import-source gate and the third-address gate, and **zero**
  before: `*` is not a name the re-export set holds, so matching each alias
  against that set reported nothing while the statement bound all eight
- `from app.stt.routing import *` appended to the same file -- **one** test,
  the third-address gate, and **zero** before, for the same reason one dot
  along. It is reported as an unreadable shape rather than as eight names,
  because what the statement binds is whatever the other module holds
- `_lookup = stt_routing.get_provider` appended to
  `app/preferences/user_settings.py` -- **one** test, the third-address gate,
  and **zero** before. This was the blind spot the helper's own docstring
  admitted; a module-level assignment off an attribute is readable and is read
- `app/stt/shim.py` added with a `__getattr__` serving `app.stt.routing`, with
  `app/stt/__init__.py` publishing `is_model_loaded` out of it and
  `app/pipeline/router.py` then taking that name off the package -- **two**
  tests, and **zero** before. The name left the re-export set while staying on
  the package, so neither gate looked for it any more; the allowlist comparison
  does not catch it either, because `is_model_loaded` is a name no module was
  pinned as holding. The set is checked against `app/stt/routing.py`'s own
  definitions now, and the shim's `__getattr__` is an unreadable shape besides

Each list below is an allowlist, not a description: adding an entry is a
deliberate act a reviewer can see in the diff.
"""

from __future__ import annotations

import ast
import functools
import importlib
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

from tests.app_modules import APP_DIR, app_modules
from tests.conftest import assert_import_loads_no_module

_COMPOSITION_ROOT = "app.config"

_MAY_IMPORT_THE_COMPOSITION_ROOT = {
    "api/auth_middleware.py",
    "api/router.py",
    "main.py",
}

_MAY_IMPORT_THE_HTTP_BOUNDARY = {"main.py"}

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
    "api": {"auth_middleware.py", "error_handler.py", "router.py"},
    "audio": {"router.py", "dependencies.py", "scratch_router.py"},
    "core": set(),
    "embeddings": set(),
    "pipeline": {"router.py", "service.py", "upload_validation.py"},
    "preferences": {"router.py"},
    "stt": {"router.py"},
    "transcripts": {"history_router.py", "words_router.py"},
}

_WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT = {"main.py"}

_IMPORT_FREE_PACKAGE_INITS = {"api", "audio"}

_RE_EXPORT_ONLY_PACKAGE_INITS = {"stt": {"__all__"}}

_STT_PACKAGE = "app.stt"

_STT_ROUTING_MODULE = "app.stt.routing"

_MAY_DEFER_AN_STT_IMPORT = {
    "app/main.py": {"app.stt.local_setup", "app.stt.routing"},
    "app/pipeline/service.py": {"app.stt.local_setup"},
    "app/preferences/router.py": {"app.stt.local_setup"},
    "app/stt/local.py": {"app.stt.local_factory"},
    "app/stt/local_factory.py": {"app.stt.local", "app.stt.local_whisper_cpp"},
    "app/stt/router.py": {"app.stt.local_setup"},
    "app/stt/routing.py": {
        "app.stt.cloud",
        "app.stt.groq_whisper",
        "app.stt.local_factory",
    },
}

_MAY_REACH_THE_ROUTING_LAYER_THROUGH_THE_PACKAGE = frozenset()

_MAY_HOLD_A_ROUTING_NAME = {
    "app/pipeline/service.py": {"get_routed_provider", "is_local_provider"},
    "app/stt/router.py": {"clear_cache", "get_provider"},
}

_KNOWN_TWO_NODE_PACKAGE_CYCLES = {
    ("app.preferences", "app.stt"),
}

_MUTUALLY_DEPENDENT_PACKAGES = frozenset(
    {
        frozenset(
            {
                "app.embeddings",
                "app.preferences",
                "app.stt",
                "app.transcripts",
            }
        ),
    }
)

_NON_FEATURE_PACKAGES = {"core"}

_HTTP_BOUNDARY_PACKAGES = {"api"}

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
    for relative, path in app_modules():
        parts = ["app", *Path(relative).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return MappingProxyType(found)


@functools.cache
def _tree(path: Path) -> ast.Module:
    """One module's parsed tree, read and parsed once per session.

    `app/stt/__init__.py` alone was read and parsed four times per run -- once
    for its `__all__`, once for its bound names, once for the re-export block
    the routing gates read, and once by the re-export gate's own walk -- and
    the two whole-tree gates re-parse every module under `app/` on top of the
    walks already here. Cached like `_modules()` and for the same reason: one
    object is shared between call sites, and nothing below mutates a tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"))


def _containing_package(path: Path) -> str:
    parts = list(path.relative_to(APP_DIR.parent).with_suffix("").parts)
    return ".".join(parts[:-1])


def _import_from_base(node: ast.ImportFrom, package: str) -> str:
    """The module one `from ... import ...` takes its names out of.

    The base on its own, resolved through the relative spelling: `from
    .routing import clear_cache` written inside `app/stt/` and `from
    app.stt.routing import clear_cache` written anywhere both answer
    `app.stt.routing`, and `from . import routing` answers `app.stt`, which is
    the package those names are taken off rather than the module one of them
    happens to name.

    `_import_from_names` returns this and every submodule the statement names
    beside it, which is what the package-level rules match on. A rule asking
    which module a name *came from* needs the base alone: `from app.stt import
    routing, clear_cache` has `app.stt.routing` among its resolved names while
    taking `clear_cache` off the package, and reading that as "this statement
    imports from the routing module" is the substitution two of this file's
    gates were built on.
    """
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    parts = parts[: max(len(parts) - node.level + 1, 0)]
    if node.module:
        parts = parts + node.module.split(".")
    return ".".join(parts)


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
    base = _import_from_base(node, package)
    if node.level and not node.module:
        return [f"{base}.{alias.name}" if base else alias.name for alias in node.names]
    bases = [base] if base else []

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
    tree = _tree(path)
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


def _package_modules(package: str) -> dict[str, Path]:
    """Every `.py` file under `app/<package>/`, keyed by its package-relative path.

    Two rules match a module against that package's exempt set and both must
    spell the key the same way, so the key is cut from the shared walk's own
    spelling rather than derived a second time from a sub-root of it.
    """
    prefix = f"{package}/"
    return {
        relative[len(prefix) :]: path
        for relative, path in app_modules()
        if relative.startswith(prefix)
    }


def _reaches_the_http_boundary(imported: str) -> bool:
    """Whether one imported dotted name names the HTTP boundary or a module of it.

    A prefix match, like the composition root's, and the one spelling every
    boundary rule shares: `app.api` and `app.api.router` both answer True, so a
    submodule cannot walk past a rule written for the package.
    """
    return any(
        imported == f"app.{package}" or imported.startswith(f"app.{package}.")
        for package in _HTTP_BOUNDARY_PACKAGES
    )


def _reaches_the_composition_root(imported: str) -> bool:
    """Whether one imported dotted name names the composition root.

    A prefix match, like the downward check's, rather than the equality this
    started as: `app/config.py` is a module today, and the day it becomes
    `app/config/` an equality test stops matching `app.config.runtime` and goes
    silently slack.
    """
    return imported == _COMPOSITION_ROOT or imported.startswith(f"{_COMPOSITION_ROOT}.")


def test_no_shared_ground_module_imports_a_feature_package_or_the_http_boundary():
    """Shared ground is the layer every package may import, so it must not
    import them back, and nothing in it is exempt from that.

    Walked over `_NON_FEATURE_PACKAGES` rather than over the name `core`, so a
    package refiled as shared ground is measured against this rule instead of
    leaving it. The walk covers the HTTP boundary as well as the feature
    packages: filing `api` as shared ground would otherwise let a `core` module
    import a router with every gate in this file green (ADR 082)."""
    shared = {f"app.{package}" for package in _NON_FEATURE_PACKAGES}
    forbidden = _FEATURE_PACKAGES | _HTTP_BOUNDARY_PACKAGES
    checked = []
    reaching_down = []
    for module, path in _modules().items():
        if _package_of(module) not in shared:
            continue
        checked.append(module)
        for imported in _imported_names(path):
            head = imported.split(".")
            if len(head) >= 2 and head[0] == "app" and head[1] in forbidden:
                reaching_down.append(f"{module} -> {imported}")

    assert checked, (
        "the walk found no shared-ground module at all, so this rule checks "
        f"nothing: _NON_FEATURE_PACKAGES is {sorted(_NON_FEATURE_PACKAGES)} "
        "(ADR 079)."
    )
    assert not reaching_down, (
        "These shared-ground modules import a feature package or the HTTP "
        f"boundary: {reaching_down}. There is no exception list any more: the "
        "module belongs outside shared ground (see docs/style-guide.md §1a), "
        "or the settings class it wants belongs in the package that reads it "
        "(ADR 073)."
    )


def test_only_the_module_that_builds_the_application_imports_the_http_boundary():
    """The HTTP boundary is composed by the module that builds the FastAPI
    application and by nothing else, so a feature package cannot acquire the
    app's middleware and exception wiring by importing it.

    Scoped to every module under `app/` rather than to the feature packages,
    because the spelling is plantable anywhere, and it prefix-matches so a
    submodule of the boundary cannot walk past. Its own function rather than a
    second assertion in the rule above: the first `assert` to fire hides the
    second (ADR 082)."""
    reaching_in = []
    for module, path in _modules().items():
        if path.relative_to(APP_DIR).as_posix() in _MAY_IMPORT_THE_HTTP_BOUNDARY:
            continue
        if _reaches_the_http_boundary(_package_of(module)):
            continue
        for imported in _imported_names(path):
            if _reaches_the_http_boundary(imported):
                reaching_in.append(f"{module} -> {imported}")

    assert _HTTP_BOUNDARY_PACKAGES, (
        "the rule matched every import against an empty boundary, so nothing "
        "could offend it: _HTTP_BOUNDARY_PACKAGES is empty and this gate "
        "checks nothing (ADR 079)."
    )
    assert not reaching_in, (
        f"These modules import the HTTP boundary: {reaching_in}. Routes, "
        "middleware and exception handlers are composed by app/main.py alone; "
        "a package that wants one of them wants a primitive under app/core/ "
        "instead. Adding a second consumer to _MAY_IMPORT_THE_HTTP_BOUNDARY is "
        "a decision, not a formality — see ADR 082."
    )


def test_no_http_boundary_module_imports_a_feature_package():
    """The boundary sits above the feature packages and below nothing, so it
    reaches `app.core` and no further. A router that imports a feature package
    puts the tangle this spec removed back one directory along.

    The walk pins itself non-empty: emptying `_HTTP_BOUNDARY_PACKAGES` is the
    mutation that would make this rule green while deleting it (ADR 079)."""
    boundary = {f"app.{package}" for package in _HTTP_BOUNDARY_PACKAGES}
    checked = []
    reaching_across = []
    for module, path in _modules().items():
        if _package_of(module) not in boundary:
            continue
        checked.append(module)
        for imported in _imported_names(path):
            head = imported.split(".")
            if len(head) >= 2 and head[0] == "app" and head[1] in _FEATURE_PACKAGES:
                reaching_across.append(f"{module} -> {imported}")

    assert checked, (
        "the walk found no HTTP-boundary module at all, so this rule checks "
        f"nothing: _HTTP_BOUNDARY_PACKAGES is {sorted(_HTTP_BOUNDARY_PACKAGES)}."
    )
    assert not reaching_across, (
        f"These HTTP-boundary modules import a feature package: "
        f"{reaching_across}. The boundary may reach app/core/ and nothing "
        "else; a route that needs a feature package belongs in that package's "
        "own router (see docs/style-guide.md §1a)."
    )


def test_every_http_boundary_module_is_reached_from_the_boundary():
    """A boundary module nothing in the boundary imports is not boundary code.

    Reachability from the module that builds the application, followed through
    the package, rather than an exempt-set entry: boundary code is free to
    import no web framework — a pydantic response model is the shape that
    arrives — and an inventory over that set left such a module no
    configuration this file accepts (ADR 082)."""
    modules = _modules()
    pending = [
        name
        for path in modules.values()
        if path.relative_to(APP_DIR).as_posix() in _MAY_IMPORT_THE_HTTP_BOUNDARY
        for name in _imported_names(path)
        if _reaches_the_http_boundary(name)
    ]
    reached = set()
    while pending:
        current = pending.pop()
        if current in reached or current not in modules:
            continue
        reached.add(current)
        pending.extend(
            name
            for name in _imported_names(modules[current])
            if _reaches_the_http_boundary(name)
        )

    checked = []
    orphans = []
    for module, path in sorted(modules.items()):
        if path.name == "__init__.py":
            continue
        if not _reaches_the_http_boundary(_package_of(module)):
            continue
        checked.append(module)
        if module not in reached:
            orphans.append(module)

    assert checked, (
        "the walk found no HTTP-boundary module at all, so this rule checks "
        f"nothing: _HTTP_BOUNDARY_PACKAGES is {sorted(_HTTP_BOUNDARY_PACKAGES)} "
        "and the directories it names hold no module (ADR 079)."
    )
    assert not orphans, (
        f"Nothing in the HTTP boundary imports these modules of it: {orphans}. "
        "A module here is reached by app/main.py or by another boundary module; "
        "one that is reached by neither is not routes, middleware or an "
        "exception handler and belongs in the package that reads it (ADR 082)."
    )


def test_only_the_recorded_modules_import_the_composition_root_directly():
    """The aggregate is for application assembly and the HTTP boundary, and
    `_MAY_IMPORT_THE_COMPOSITION_ROOT` is the whole of who may reach it.

    Its own function rather than a second assertion inside the rule above,
    because the first `assert` to fire hides the second: a diff that both plants
    a feature-package import in `core` and reaches for `app.config` elsewhere
    would report only the half that ran first.

    Scoped to every module under `app/`, not to `core` alone. `from app.config
    import settings` is plantable in any package, and scoping the walk to `core`
    left it green everywhere else — which is what it did when this gate shipped
    (ADR 091)."""
    reaching_up = []
    for module, path in _modules().items():
        relative = path.relative_to(APP_DIR).as_posix()
        if relative in _MAY_IMPORT_THE_COMPOSITION_ROOT:
            continue
        for imported in _imported_names(path):
            if _reaches_the_composition_root(imported):
                reaching_up.append(f"{module} -> {imported}")

    assert not reaching_up, (
        f"These modules import {_COMPOSITION_ROOT} directly: {reaching_up}. "
        "Read the settings instance the package owns — `stt_settings`, "
        "`audio_settings`, `embedding_settings` — not the aggregate. Adding a "
        "name to _MAY_IMPORT_THE_COMPOSITION_ROOT is a decision, not a "
        "formality — see ADR 091."
    )


def test_the_composition_root_gate_still_matches_something():
    """The mirror the two allowlists below get, for the two constants above.

    The gate matches a dotted string against imports and a path against the
    tree, so renaming or moving either end leaves it matching nothing at all —
    green, with every assertion in it vacuous. That is not hypothetical: before
    this test existed, renaming `app/config.py` to `app/composition.py` and
    repointing its importers left all fourteen tests in this file passing."""
    stale = []
    if _COMPOSITION_ROOT not in _modules():
        stale.append(f"{_COMPOSITION_ROOT}: no such module")
    for relative in sorted(_MAY_IMPORT_THE_COMPOSITION_ROOT):
        path = APP_DIR / relative
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
        "singleton is now defined and at who still reads it — until then the "
        "test above passes without checking anything."
    )


def test_the_http_boundary_gate_still_matches_something():
    """The mirror the composition root's two constants get, for the boundary's.

    Every rule over the boundary matches a package name against the tree and a
    path against an allowlist, so a renamed package, or an entry written ahead
    of the import it covers, leaves those rules green with nothing matched."""
    stale = []
    for package in sorted(_HTTP_BOUNDARY_PACKAGES):
        if not (APP_DIR / package).is_dir():
            stale.append(f"{package}: no such package")
    for relative in sorted(_MAY_IMPORT_THE_HTTP_BOUNDARY):
        path = APP_DIR / relative
        if not path.exists():
            stale.append(f"{relative}: no such module")
        elif not any(
            _reaches_the_http_boundary(imported) for imported in _imported_names(path)
        ):
            stale.append(f"{relative}: imports no HTTP boundary module")

    assert _MAY_IMPORT_THE_HTTP_BOUNDARY, (
        "the mirror walked an empty allowlist, so no entry could be stale: "
        "_MAY_IMPORT_THE_HTTP_BOUNDARY is empty (ADR 079)."
    )
    assert not stale, (
        f"The HTTP-boundary gates describe nothing: {stale}. Repoint "
        "_HTTP_BOUNDARY_PACKAGES at the package the routes, middleware and "
        "exception handlers live in, and _MAY_IMPORT_THE_HTTP_BOUNDARY at the "
        "module that composes them — an entry covering no import hands the "
        "next module to take that path a free pass (ADR 082)."
    )


def test_the_base_dsp_module_imports_nothing_the_sidecar_lacks():
    """ADR 015 rests on `audio/analysis.py` staying inside what the frozen
    PyInstaller sidecar actually ships — numpy, soundfile, sounddevice. Its own
    docstring names the libraries that would break the packaged build. A
    violation ships broken rather than failing here, and only a tag push would
    reveal it."""
    offenders = []
    for relative, forbidden in _SIDECAR_ABSENT_LIBRARIES.items():
        path = APP_DIR / relative
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
        path = APP_DIR / relative
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
    for path in sorted((APP_DIR / "audio").glob("*.py")):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "SystemAudioSource"
                for base in node.bases
            ):
                found.append(path.relative_to(APP_DIR).as_posix())
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
        for imported in _imported_names(APP_DIR / relative):
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
        for path in APP_DIR.iterdir()
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
        for relative, path in sorted(_package_modules(package).items()):
            if relative in exempt:
                continue
            if _imports_a_web_framework(path):
                offenders.append(path.relative_to(APP_DIR).as_posix())

    for path in sorted(APP_DIR.glob("*.py")):
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


def test_shared_ground_carries_no_web_framework_exemption():
    """Shared ground is what every other package may import, so not one module
    in it may reach a web framework — under any exemption, for any reason.

    The gate above accepts whatever exempt set the dict happens to carry,
    including a fresh one. Stated over the classification rather than over one
    package name, so a second package filed as shared ground inherits the rule
    instead of arriving unguarded, and read through `.get` so a deleted key
    reports this message rather than a `KeyError` (ADR 082)."""
    checked = sorted(_NON_FEATURE_PACKAGES)
    offenders = [
        package
        for package in checked
        if _WEB_FRAMEWORK_FREE_PACKAGES.get(package) != set()
    ]

    assert checked, (
        "the walk found no shared-ground package, so this rule checks nothing: "
        "_NON_FEATURE_PACKAGES is empty (ADR 079)."
    )
    assert not offenders, (
        f"These packages are importable from anywhere and either carry a "
        f"web-framework exemption or carry no allowlist entry at all: "
        f"{offenders}. A module that needs fastapi or starlette is "
        "HTTP-boundary code and belongs in app/api/, not in a package every "
        "other package may import."
    )


def test_the_package_walk_finds_the_directories_it_is_meant_to_check():
    """Both coverage rules over this walk pass vacuously when it is empty.

    Membership rather than a count: a count reddens on the next package anyone
    adds, which is the failure direction that gets a gate deleted, not fixed.
    """
    packages = set(_package_directories())
    assert {"api", "audio", "core", "stt", "transcripts", "preferences"} <= packages


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
        if not (APP_DIR / package).is_dir():
            stale.append(f"{package}: no such package")
            continue
        modules = _package_modules(package)
        for relative in sorted(exempt):
            if relative not in modules:
                stale.append(f"{package}/{relative}: no such module")
            elif not _imports_a_web_framework(modules[relative]):
                stale.append(f"{package}/{relative}: imports no web framework")

    for name in sorted(_WEB_FRAMEWORK_FREE_APP_ROOT_EXCEPT):
        path = APP_DIR / name
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


def test_a_docstring_only_package_surface_holds_nothing_else():
    """A package `__init__.py` executes on every `app.<package>.<module>`
    import, so anything it holds is paid for by every consumer. `app.audio`
    once re-exported both recorders, which made the pure numpy module
    `app.audio.timeline` drag the whole capture stack behind it.

    `app.api` is in the set for a different reason: the package surface is half
    of what holds the HTTP boundary to boundary code, the per-module exempt set
    being the other half, and a re-export there would hand any importer the
    whole boundary under one name that no exempt set covers (ADR 082).

    Checked as "nothing but a docstring" rather than "no import statements",
    because a module-level `__getattr__` restores the same re-export while
    leaving the import statements absent — it defers the cost to the first
    attribute read instead of removing it, and the runtime test below cannot
    see it either, since importing a submodule never invokes it."""
    offenders = []
    for package in sorted(_IMPORT_FREE_PACKAGE_INITS):
        path = APP_DIR / package / "__init__.py"
        assert path.exists(), (
            f"{package}/__init__.py no longer exists — deleting it turns "
            f"{package} into a namespace package, which changes the pinned "
            "property rather than satisfying it. Update this test."
        )
        body = list(_tree(path).body)
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


def _bare_target_names(target: ast.expr) -> list[str] | None:
    """The names a bare assignment target binds, or None when it binds none.

    `x` binds one and `x, (y, z)` binds three. A subscript or an attribute
    target -- `__all__[0]`, `obj.field` -- binds nothing: it calls a method on
    an object the module already holds, and the name it appears to mention is
    being read rather than written. None says so, and the caller reads it as
    "this statement is not a declaration"."""
    if isinstance(target, ast.Name):
        return [target.id]
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    names: list[str] = []
    for element in target.elts:
        nested = _bare_target_names(element)
        if nested is None:
            return None
        names.extend(nested)
    return names


def _module_level_assigned_names(node: ast.stmt) -> list[str]:
    """Every name one statement declares, in the two spellings that declare
    one -- `x = ...` and `x: T = ...`. Anything else comes back empty, which
    the allowlist below reads as an offender rather than as a free pass.

    Walking the whole target for every `ast.Name` reported `__all__` for
    `__all__[0] = threading.Lock()`, which binds nothing and runs a call
    instead, and `x += ...` reported the name it mutates in place. Both
    matched an allowlist written for the statement that publishes the
    surface, and a re-export surface is published once by a plain
    assignment."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    names: list[str] = []
    for target in targets:
        bare = _bare_target_names(target)
        if bare is None:
            return []
        names.extend(bare)
    return sorted(set(names))


def _all_declarations(package: str) -> list[ast.stmt]:
    """Every module-level statement in a package `__init__.py` that assigns
    `__all__`, in file order.

    Every one of them rather than the first. Stopping at the first read the
    surface as it was one statement earlier, so a second `__all__ = __all__ +
    ["Bogus"]` published a name nothing binds and `from app.stt import *`
    raised `AttributeError: module 'app.stt' has no attribute 'Bogus'` with
    this file reporting 22 passed."""
    return [
        node
        for node in _tree(APP_DIR / package / "__init__.py").body
        if _module_level_assigned_names(node) == ["__all__"]
    ]


def _names_published_by(node: ast.stmt) -> frozenset[str] | None:
    """The names one `__all__` assignment publishes, or None when its value is
    not a literal sequence of strings.

    None rather than an empty set, because the two mean opposite things to a
    caller: `__all__ = _computed()` publishes a surface no walk here can read,
    and reading it as "publishes nothing" would make every import beside it an
    offender while making a widened surface invisible. The gate below reports
    the statement instead."""
    value = getattr(node, "value", None)
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    if not all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in value.elts
    ):
        return None
    return frozenset(element.value for element in value.elts)


def _package_init_published_names(package: str) -> frozenset[str]:
    """Every name a package `__init__.py` lists across its module-level
    `__all__` assignments.

    Read off the tree rather than by importing the package, the way every
    other static rule in this module reads it: importing `app.stt` to ask for
    its `__all__` would run the very import block being judged. The runtime
    surface is read by `test_the_stt_package_publishes_every_name_it_promises`
    instead, which is where a value this walk cannot follow gets caught."""
    names: set[str] = set()
    for node in _all_declarations(package):
        published = _names_published_by(node)
        if published is not None:
            names |= published
    return frozenset(names)


def _package_init_imported_names(package: str) -> dict[str, str]:
    """Every name a package `__init__.py` binds by importing it, mapped to the
    module the import names.

    The bound name is the alias wherever one is written, because the alias is
    what a consumer reads off the package: `from app.stt.routing import
    clear_cache as drop_cache` publishes `drop_cache` and nothing called
    `clear_cache`.

    The module half is what `test_the_stt_package_publishes_every_name_it_
    promises` compares the published object against, and it is the half that
    makes an alias visible at all: the bound name is all the re-export gate
    reads, so `from app.stt.routing import get_routed_provider as
    get_provider` satisfied it while making `app.stt.get_provider` a different
    function from `app.stt.routing.get_provider` -- 22 passed, and
    `tests/test_factories.py:5` takes `get_provider` off the package."""
    path = APP_DIR / package / "__init__.py"
    dotted = f"app.{package}"
    bound: dict[str, str] = {}
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom):
            resolved = _import_from_names(node, dotted)
            source = resolved[0] if resolved else ""
            for alias in node.names:
                bound[alias.asname or alias.name] = source
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = alias.name
    return bound


def _is_a_re_export_of(node: ast.stmt, package: str, published: frozenset[str]) -> bool:
    """Whether one statement imports a sibling module's name for this package
    to publish.

    `from app.stt.routing import clear_cache` is one while `clear_cache` is in
    `__all__`. Four things are not, and the arm this replaces accepted every
    one of them, because it skipped `ast.Import` and `ast.ImportFrom` on
    sight: a plain `import json, socket, sqlite3`, which binds three names the
    package never publishes; `from app.core.gpu_probe import GpuVendor`, which
    reaches out of the package rather than re-exporting inside it; `from
    app.stt.local_whisper_cpp import WhisperCppServerSTTProvider`, which drags
    the whisper.cpp provider stack into every `import app.stt.<anything>` and
    publishes nothing -- reproduced against the old arm, 21 passed; and `from
    app.stt.routing import *`, which names nothing it could publish.

    The *bound* name is all this reads, which is exactly what it can decide
    from a tree: `from app.stt.routing import get_routed_provider as
    get_provider` binds a published name and is accepted here. Whether that
    name still refers to the object its own module holds is a question about
    the running package, and
    `test_the_stt_package_publishes_every_name_it_promises` asks it there.

    `tests/test_settings_isolation.py`'s
    `test_the_core_config_module_holds_nothing_but_its_re_export` records the
    identical unconditional skip in its own docstring, and this is its shape:
    accept the re-export the file is for, by name, and report everything
    else."""
    if not isinstance(node, ast.ImportFrom):
        return False
    dotted = f"app.{package}"
    sources = _import_from_names(node, dotted)
    if not sources or not all(source.startswith(f"{dotted}.") for source in sources):
        return False
    return bool(node.names) and all(
        (alias.asname or alias.name) in published for alias in node.names
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
    implements, and the docstring-strip idiom below was taken from it.

    The import arm is an allowlist too, and shipped as the same unconditional
    skip that file's docstring records having removed from itself: every
    `ast.Import` and `ast.ImportFrom` was waved through on node type alone, so
    `from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider` beside
    `import json, socket, sqlite3` left this file reporting clean while every
    `import app.stt.<anything>` paid for the whisper.cpp provider stack --
    reproduced, 21 passed. `_is_a_re_export_of` decides it now: a sibling
    module inside this package, and every name it binds published in `__all__`.
    The other direction -- a name `__all__` promises that no import binds --
    belongs to `test_the_stt_package_publishes_every_name_it_promises`, so the
    two halves fail separately and each failure names its own cause.

    The allowlist matches a name only where the statement declares it. A
    target that mentions `__all__` without binding it -- `__all__[0] =
    threading.Lock()` -- used to arrive here as the string `__all__` and be
    waved through while running a call, and so did `__all__ += _something`.
    Both come back with no declared name now and are reported by node
    type.

    The surface is also published once, from a list of string literals. A
    second `__all__ = __all__ + ["Bogus"]` is an ordinary assignment binding
    an allowed name, so the allowlist accepted it while the reader above
    returned on the first `__all__` and never saw the widened surface; the
    value is a `BinOp`, so collecting every assignment would not have seen it
    either. A repeat of an allowed name, and an `__all__` whose value is not a
    literal sequence of names, are both reported here, and the runtime half is
    property 12."""
    offenders = []
    for package, allowed in sorted(_RE_EXPORT_ONLY_PACKAGE_INITS.items()):
        path = APP_DIR / package / "__init__.py"
        assert path.exists(), (
            f"{package}/__init__.py no longer exists — deleting it turns "
            f"{package} into a namespace package, which drops the re-export "
            "surface this pins rather than satisfying it. Update this test."
        )
        body = list(_tree(path).body)
        if body and ast.get_docstring(ast.Module(body=body, type_ignores=[])):
            body = body[1:]
        published = _package_init_published_names(package)
        declared: set[str] = set()
        for node in body:
            if _is_a_re_export_of(node, package, published):
                continue
            assigned = _module_level_assigned_names(node)
            if assigned and all(name in allowed for name in assigned):
                repeated = sorted(set(assigned) & declared)
                declared.update(assigned)
                if repeated:
                    offenders.append(
                        f"{package}/__init__.py:{node.lineno} assigns "
                        f"{', '.join(repeated)} a second time"
                    )
                elif "__all__" in assigned and _names_published_by(node) is None:
                    offenders.append(
                        f"{package}/__init__.py:{node.lineno} assigns __all__ "
                        "something other than a list of names"
                    )
                continue
            described = (
                f"assigns {', '.join(assigned)}" if assigned else _statement_description(node)
            )
            offenders.append(f"{package}/__init__.py:{node.lineno} {described}")

    assert not offenders, (
        f"These re-export surfaces hold something that is not a re-export: "
        f"{offenders}. A package `__init__.py` listed in "
        "_RE_EXPORT_ONLY_PACKAGE_INITS holds a docstring, `__all__`, and "
        "imports that publish exactly the names `__all__` lists, taken from "
        "modules inside the package — nothing else, in any position, "
        "including inside an `if` or a `try` block. `__all__` is assigned "
        "once, from a list of string literals: a second assignment widens the "
        "surface past what any import binds, and a computed one is a surface "
        "no walk here can read. An import binding a name "
        "the package does not publish is a dependency every consumer of every "
        "submodule pays for; put it in the module that needs it. Behaviour "
        "belongs in a sibling module, which is what `app/stt/routing.py` is. "
        "See docs/style-guide.md §1a."
    )


def test_the_stt_package_publishes_every_name_it_promises():
    """`__all__`, the import block above it and the objects the package ends
    up holding are three views of one surface, and nothing read them against
    each other.

    Read off the imported package rather than off its tree, which is what
    makes this the outcome check the three static gates around it are not.
    Three review rounds each found a different *spelling* walking past a walk
    written for the previous one -- a bare relative import, a submodule named
    in the same statement, an unconditional skip by node type -- and this asks
    what `app.stt` actually holds instead. A second `__all__ = __all__ +
    ["Bogus"]` is a `BinOp` no `elts` walk can read, and the package answers
    with `Bogus` in `__all__` and no attribute of that name; an aliased
    re-export satisfies every name-level check and answers with a different
    function. Both reproduced against the static gates alone, 22 passed.

    Identity, name by name: `app.stt.<name> is <the module it was imported
    from>.<name>`. `from app.stt.routing import get_routed_provider as
    get_provider` binds a published name to a second object -- measured,
    `app.stt.get_provider.__name__` is `get_routed_provider`, whose signature
    is `(stt_settings, audio_duration, file_extension)` rather than
    `(mode, stt_settings)`, so `tests/test_factories.py:5` takes a function
    that cannot be called the way it calls it. The docstring this replaces
    claimed the property held structurally and needed no assertion; it did not
    hold, and this is the assertion.

    The import block is what names the module each published name is compared
    against, which is all the static half of this gate decides: the walk picks
    the names, the running package answers for them."""
    package = importlib.import_module(_STT_PACKAGE)
    published = sorted(getattr(package, "__all__", []))
    assert published, (
        f"{_STT_PACKAGE} declares no `__all__`, so this gate is matching an "
        "empty surface and would pass against any import block at all. "
        "Restore it, or drop `stt` from _RE_EXPORT_ONLY_PACKAGE_INITS and "
        "delete this test."
    )

    imported = _package_init_imported_names("stt")
    unbound = sorted(name for name in published if name not in imported)

    assert not unbound, (
        f"app/stt/__init__.py promises {unbound} in `__all__` and imports "
        "nothing that binds them, so `from app.stt import *` raises "
        "AttributeError and `from app.stt import <name>` raises ImportError, "
        "for every consumer, at import time. Import each name from the module "
        "that defines it, or take it out of `__all__`. Names the file "
        f"actually binds: {sorted(imported)}."
    )

    rebound = []
    for name in published:
        source_name = imported[name]
        try:
            source = importlib.import_module(source_name)
        except (ImportError, ValueError):
            rebound.append(f"{name} is published from {source_name!r}, not a module")
            continue
        if not hasattr(package, name):
            rebound.append(f"{_STT_PACKAGE}.{name} does not exist")
        elif not hasattr(source, name):
            rebound.append(f"{source_name}.{name} does not exist")
        elif getattr(package, name) is not getattr(source, name):
            rebound.append(f"{_STT_PACKAGE}.{name} is not {source_name}.{name}")

    assert not rebound, (
        f"These published names are not the objects their own modules hold: "
        f"{rebound}. A consumer reaching `{_STT_PACKAGE}.<name>` and a module "
        "reaching `<source>.<name>` must get one object, or a "
        "`monkeypatch.setattr` on either address leaves the other running "
        "unpatched code and the signatures are free to diverge. Re-export "
        "each name under the name its module gives it, with no `as`, and "
        "assign nothing in that file but `__all__`."
    )


def _function_body_imports(path: Path) -> list[ast.stmt]:
    """Every import statement written inside a function body, deduplicated.

    A function nested in another function is reached twice by the outer walk,
    so the statements are keyed by identity rather than appended blindly; a
    duplicate would report the same site twice and make the message read as two
    defects."""
    tree = _tree(path)
    found: dict[int, ast.stmt] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Import, ast.ImportFrom)):
                found[id(inner)] = inner
    return sorted(found.values(), key=lambda statement: statement.lineno)


def _reaches_the_stt_package(name: str) -> bool:
    """Whether one resolved dotted name is the `app.stt` package or something
    inside it.

    The package and its modules both, which is what makes the gate below
    survive the routing layer moving out of `__init__.py`: the names it pins
    live in `app.stt.routing` now, so matching the package alone would let
    every one of them be deferred again under a spelling nobody had to
    invent. `app.stt_extra` is a different package and the dot is what says
    so."""
    return name == _STT_PACKAGE or name.startswith(f"{_STT_PACKAGE}.")


def _deferred_stt_imports(path: Path, package: str) -> set[str]:
    """Every `app.stt` name one module imports from inside a function body.

    Resolved names rather than source lines, so the pin below does not churn
    when a function moves. Each spelling arrives as the same string it would
    have written absolutely: `from . import clear_cache` inside `app/stt/`
    resolves to `app.stt.clear_cache`, `from .. import stt` outside it to
    `app.stt`, and `from app.stt import local_whisper_cpp_cmd` to both the
    package and the submodule it names."""
    found: set[str] = set()
    for node in _function_body_imports(path):
        if isinstance(node, ast.ImportFrom):
            names = _import_from_names(node, package)
        else:
            names = [alias.name for alias in node.names]
        found.update(name for name in names if _reaches_the_stt_package(name))
    return found


def _routing_module_public_definitions() -> frozenset[str]:
    """Every public name `app/stt/routing.py` defines at module level.

    Definitions, not imports: `STTProvider` reaches that module from
    `app.stt.base` and stays that module's name wherever it is read, so a rule
    forbidding a second address for it would forbid the first one.

    What it is for is the guard in `_stt_routing_re_exports`. Both routing
    gates measure against the names the package re-exports, and that set is
    read off one import block; a name published out of a third module inside
    the package would leave the block, leave the set, and stay on the package,
    with both gates still non-empty and so still green."""
    found: set[str] = set()
    for node in _tree(_modules()[_STT_ROUTING_MODULE]).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            found.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return frozenset(name for name in found if not name.startswith("_"))


def _stt_routing_re_exports() -> frozenset[str]:
    """The names `app/stt/__init__.py` publishes out of `app.stt.routing`.

    Read off that file rather than listed here, so the gates below cannot fall
    behind the surface they pin: a name added to the re-export block is covered
    the moment it is written. Which statements count is resolved through
    `_import_from_base`, the way every other rule in this file resolves an
    import. It compared `node.module` against the dotted string for two review
    rounds, and one `from .routing import peek_local_provider` took that name
    out of the set both gates measure against while leaving the set non-empty,
    so neither gate's `assert re_exported` tripwire fired and the name was free
    at a third address.

    Selecting the names is where both gates can be emptied without either one
    reporting anything, so two shapes are a failure here rather than a miss.
    `from app.stt.routing import *` names nothing this can enumerate. A name
    published out of a third module that serves routing's object -- `from
    app.stt.shim import is_model_loaded` -- leaves the block while staying on
    the package, which is why the set is checked against what
    `app/stt/routing.py` actually defines rather than trusted on its own."""
    path = APP_DIR / "stt" / "__init__.py"
    package = _containing_package(path)
    statements = [
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and _import_from_base(node, package) == _STT_ROUTING_MODULE
    ]
    starred = [
        node.lineno
        for node in statements
        if any(alias.name == "*" for alias in node.names)
    ]
    assert not starred, (
        f"app/stt/__init__.py re-exports {_STT_ROUTING_MODULE} with a star "
        f"import at line(s) {starred}, so the names it publishes are whatever "
        "that module happens to hold and this walk can list none of them. "
        "Both routing gates would then measure against a set missing every "
        "starred name. Spell each re-export out."
    )

    re_exported = frozenset(alias.name for node in statements for alias in node.names)
    expected = _routing_module_public_definitions() & set(
        _package_init_imported_names("stt")
    )
    assert re_exported == expected, (
        f"app/stt/__init__.py publishes {sorted(expected)} of "
        f"{_STT_ROUTING_MODULE}'s own definitions and this walk sees "
        f"{sorted(re_exported)} of them coming from that module. A routing "
        "name the package publishes through anything else -- a third module "
        "inside the package, or an `as` that renames it -- drops out of the "
        "set both routing gates measure against while staying reachable as "
        f"`{_STT_PACKAGE}.<name>`, and both gates stay green on a surface "
        "they no longer cover. Import each one straight from "
        f"`{_STT_ROUTING_MODULE}`, under the name that module gives it."
    )
    return re_exported


def _routing_names_taken_from_the_package(
    node: ast.stmt, package: str, re_exported: frozenset[str]
) -> list[str]:
    """Which routing names one import statement takes from the `app.stt`
    package instead of from `app.stt.routing`.

    `from app.stt import clear_cache` does, and so does the bare relative
    `from . import clear_cache` written inside the package. `from
    app.stt.routing import clear_cache` does not, and neither does `from
    app.stt import local_whisper_cpp_cmd`, which names a sibling module the
    package does not re-export.

    Binding the package object itself -- `import app.stt`, `from app import
    stt` -- counts as taking all of them, because the only reason to hold that
    object is to read a name off it, and an attribute chain is past what this
    walk reads. `from app.stt import *` counts as taking all of them for the
    opposite reason: it binds every re-exported name at once, and matching
    each alias against the re-export set read `*` as a name nobody publishes
    and reported nothing. `import app.stt.local_setup` counts for the same reason and
    matched nothing while this arm compared for equality: it binds `app` just
    as the shorter spelling does, so `app.stt.clear_cache()` planted beside it
    in `app/pipeline/service.py` left the file at 22 passed. It prefix-matches
    through `_reaches_the_stt_package`, the way the deferral gate two
    functions up already did.

    Every imported name is judged on its own and the statement carries no
    verdict of its own, which is the correction this arrived without. Whether
    `app.stt.routing` appeared anywhere among the statement's resolved names
    used to answer for the whole statement, and `from app.stt import routing,
    get_routed_provider, is_local_provider` resolves to the module *and* the
    package: the early return read the module, waved the statement through,
    and `get_routed_provider` bound off the package -- the second address this
    exists to forbid -- with every gate in this file green. Reproduced at
    `app/pipeline/service.py:20`, 21 passed. Naming a submodule beside a
    re-exported name is ordinary Python and was the one spelling nobody had to
    invent; `from app.stt.routing import clear_cache` needs no early return to
    pass, because neither `app.stt` nor `app.stt.clear_cache` is among its
    resolved names."""
    if isinstance(node, ast.Import):
        return (
            [_STT_PACKAGE]
            if any(_reaches_the_stt_package(alias.name) for alias in node.names)
            else []
        )
    if not isinstance(node, ast.ImportFrom):
        return []
    names = set(_import_from_names(node, package))
    leaf = _STT_PACKAGE.rsplit(".", 1)[-1]
    takes_the_whole_surface = any(alias.name in (leaf, "*") for alias in node.names)
    if _STT_PACKAGE in names and takes_the_whole_surface:
        return [_STT_PACKAGE]
    return sorted(
        alias.name
        for alias in node.names
        if alias.name in re_exported
        and (_STT_PACKAGE in names or f"{_STT_PACKAGE}.{alias.name}" in names)
    )


def test_every_function_body_import_of_the_stt_package_is_a_recorded_one():
    """`docs/style-guide.md`: a function-local `from app.…` is a cycle being
    hidden and should be read as a defect. Seven sites in three files deferred
    an `app.stt` import while the package held the routing layer; six of them
    had nowhere else to go, because the names they wanted existed only in
    `__init__.py`.

    The package *and* its modules, which is the whole point of pinning it
    after that move. Matching `app.stt` alone meant `from app.stt.routing
    import get_provider, peek_local_provider` re-deferred into
    `ensure_local_ready()` walked straight past -- the same eight names, the
    same defect, one dot further along. What caught it instead was 27
    incidental `AttributeError`s from `monkeypatch` targets, which vanish the
    first time those tests are re-pointed.

    Scoped to `app.stt` all the same, not to the general rule it quotes. A
    deferral naming another package -- `from app.core.gpu_probe import
    GpuVendor` -- walks past this gate and is meant to: catching every
    function-local `from app.…` needs a reason recorded per site, and six
    survive in this package's own files, every one of them naming `app.core`.

    Covering the modules means the allowlist stops being one filename: four
    files defer `app.stt.local_setup` to reach the prewarm entry points,
    `app/stt/routing.py` and the two platform modules defer the local factory
    and the provider classes -- four of those are the seam five test sites
    patch by string path, which a module-level import would defeat -- and
    `app/main.py` defers `app.stt.routing` for the late binding three
    `monkeypatch.setattr` sites need. It is keyed on the names each file may
    defer rather than on the file, so a file already holding one recorded
    deferral cannot acquire a second, different one for free.

    `main.py`'s entry is not a startup-time exemption, whatever the style guide
    says. Measured: `import app.main` leaves `app.stt`, `app.stt.routing` and
    `app.stt.local_setup` in `sys.modules` through the module-level
    `from app.stt.router import router` at `app/main.py:31`, so by the time
    `lifespan()` runs there is nothing left to defer. What the deferral buys is
    **late binding** -- `monkeypatch.setattr(app.stt.routing, "clear_cache",
    …)` at `tests/test_background_tasks.py:392` and `:476` and
    `tests/test_startup.py:262` replaces the attribute on the module object,
    and only a name read inside the shutdown body sees the replacement. It
    names `app.stt.routing` rather than the package because the package
    attribute is a second address for the same function, and patching one
    while production reads the other is the failure
    `test_no_module_under_app_takes_a_routing_name_off_the_stt_package` exists
    to make impossible.

    The walk is the one the cycle tests use, so a spelling that hides from this
    gate hides from those too."""
    measured = {}
    for path in sorted(_modules().values()):
        relative = path.relative_to(APP_DIR.parent).as_posix()
        deferred = _deferred_stt_imports(path, _containing_package(path))
        if deferred:
            measured[relative] = deferred

    unrecorded = {
        relative: sorted(names - _MAY_DEFER_AN_STT_IMPORT.get(relative, set()))
        for relative, names in measured.items()
        if names - _MAY_DEFER_AN_STT_IMPORT.get(relative, set())
    }
    dead = {
        relative: sorted(names - measured.get(relative, set()))
        for relative, names in _MAY_DEFER_AN_STT_IMPORT.items()
        if names - measured.get(relative, set())
    }

    assert not unrecorded and not dead, (
        f"unrecorded deferrals: {unrecorded}; entries with nothing left to "
        f"cover: {dead}. Import the module that holds the name at module "
        "level -- for the routing layer that is `app.stt.routing`, not the "
        "`app.stt` package. Adding a name to _MAY_DEFER_AN_STT_IMPORT claims "
        "the deferral buys something a module-level import cannot: late "
        "binding for a `monkeypatch.setattr` on the module object, the way "
        "`app/main.py`'s does, or a factory five test sites replace by string "
        "path. It does not buy startup time, because `app/main.py:31` has "
        "already imported the package by then."
    )


def test_no_module_under_app_takes_a_routing_name_off_the_stt_package():
    """The re-export in `app/stt/__init__.py` gives the eight routing names a
    second live address, and a second address is a second `monkeypatch`
    target. `monkeypatch.setattr("app.stt.clear_cache", ...)` replaces the
    attribute on the package; a module that bound the same function through
    `app.stt.routing` never sees it, so a test pinning "the cache was cleared
    on mode switch" can pass while asserting nothing at all.

    What this pins is the *source*, name by name: no module under `app/`
    imports a routing name from the `app.stt` package. That is one import
    spelling, not one address -- `from app.stt.routing import clear_cache`
    binds a module attribute of its own just as surely as the package
    spelling does, and this walk has nothing to say about that. Counting the
    addresses themselves is
    `test_the_routing_names_gain_no_third_address_under_app`; the two are
    separate tests because they fail for different reasons, one saying a
    module took the consumer surface and the other saying a module kept a
    copy.

    Sourcing from the package is allowed nowhere and the allowlist is empty.
    `app/main.py` was the one entry for two review rounds, on the argument
    that its shutdown body wanted the package attribute so three tests could
    replace it -- but the package is the *wrong* attribute to want, and
    wanting it was the defect rather than the reason. Binding the function
    inside the shutdown body from `app.stt.routing` buys the identical late
    binding at the address the rest of production already uses.

    Binding the module -- `from app.stt import routing`, which
    `app/preferences/user_settings.py:22` and `app/stt/local_setup.py:17` both
    do -- is not taking a name off the package and is not reported. It
    publishes no second address: every read of `routing.clear_cache` happens
    at call time and lands on the one module attribute a `monkeypatch`
    replaces. Binding the *function* off the package is the defect, whatever
    statement it rides in.

    The re-exported names are read off `app/stt/__init__.py` rather than
    listed here, so a name added to that block is covered the day it is
    written, and the walk covers module level and function bodies alike --
    `app/main.py`'s own site is a function body, and a module-level-only walk
    would have measured nothing there at all. Reading them off that block is a
    selection of its own, and `_stt_routing_re_exports` is where it is made to
    fail closed: a name this walk cannot attribute to `app.stt.routing` must
    redden that helper rather than quietly shrink the surface measured here.

    `from app.stt import *` is reported, as `import app.stt` is: it binds
    every re-exported name at once off the package, and matching `*` against
    the re-export set read it as a name nobody publishes."""
    re_exported = _stt_routing_re_exports()
    assert re_exported, (
        "app/stt/__init__.py re-exports nothing from app.stt.routing, so this "
        "gate is matching an empty set and would pass against any diff. The "
        "routing module was renamed or the re-export block was removed; "
        "update _STT_ROUTING_MODULE."
    )

    measured = {}
    for path in sorted(_modules().values()):
        relative = path.relative_to(APP_DIR.parent).as_posix()
        package = _containing_package(path)
        tree = _tree(path)
        sites = [
            f"{relative}:{node.lineno} {_statement_description(node)}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and _routing_names_taken_from_the_package(node, package, re_exported)
        ]
        if sites:
            measured[relative] = sites

    assert set(measured) == _MAY_REACH_THE_ROUTING_LAYER_THROUGH_THE_PACKAGE, (
        f"pinned:   {sorted(_MAY_REACH_THE_ROUTING_LAYER_THROUGH_THE_PACKAGE)}; "
        f"measured: {sorted(measured)}. Sites: "
        f"{sorted(site for sites in measured.values() for site in sites)}. "
        "Spell it `from app.stt.routing import ...`: the package re-export is "
        "for consumers outside `app/`, and a module reaching a routing name "
        "through it splits the `monkeypatch` surface in two. The allowlist is "
        "empty on purpose - a file added to "
        "_MAY_REACH_THE_ROUTING_LAYER_THROUGH_THE_PACKAGE claims it needs the "
        "package attribute itself, and nothing does: a shutdown body wanting "
        "late binding gets it from `app.stt.routing` at the same cost. A file "
        "leaving the measured set means its entry is now dead and should go."
    )


def _module_level_statements(tree: ast.Module) -> Iterator[ast.stmt]:
    """Every statement that runs when the module is imported.

    The top-level body, and whatever an `if`, `try`, `with`, `for` or `while`
    nests inside it, because all of those bind module attributes. A function
    or class body does not, and is not descended into. Reading `tree.body`
    alone would let `if TYPE_CHECKING: ... else: from app.stt.routing import
    clear_cache` keep a copy no walk here reports, which is the blind spot the
    re-export gate above shipped with in the other direction."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(
            child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)
        )


def _unreadable_attribute_shapes(path: Path) -> list[str]:
    """Where one module builds module attributes this file cannot enumerate.

    Two shapes, both at module level, and both of them a whole namespace
    rather than one name. `from <anything> import *` binds whatever the other
    module happens to hold, which no walk over *this* module's tree can list.
    A module-level `__getattr__` (PEP 562) answers for names that were never
    bound at all, so a routing function is served off the module with no
    statement naming it anywhere. Both spellings of it count: `def
    __getattr__` and a plain assignment binding that name to a callable
    defined elsewhere, which is the same hook and reads as ordinary code.

    Reported rather than skipped. The gate below counts the names it can see,
    and these are the two cases where seeing nothing is not the same as there
    being nothing -- which is the failure this file has now shipped five times
    in five different spellings."""
    found = []
    for node in _module_level_statements(_tree(path)):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            found.append(f"{node.lineno} star import")
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__getattr__"
        ):
            found.append(f"{node.lineno} module-level __getattr__")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(target, ast.Name) and target.id == "__getattr__"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        ):
            found.append(f"{node.lineno} module-level __getattr__")
    return found


def _routing_names_bound_at_module_level(
    path: Path, package: str, re_exported: frozenset[str]
) -> set[str]:
    """Which routing names one module keeps as an attribute of its own.

    Keyed on the name `app.stt.routing` gives the function rather than on the
    name the importer binds, because an `as` changes the spelling and not the
    address: `from app.stt.routing import get_provider as _lookup` leaves
    `<module>._lookup` holding the routing function exactly as the plain
    spelling leaves `<module>.get_provider` holding it.

    The source module is not read at all, which is the correction this
    arrived without. It skipped every statement whose resolved names led
    nowhere inside `app.stt`, so a copy taken from a module that already holds
    one -- `from app.pipeline.service import get_routed_provider` -- was
    invisible to the one gate whose entire subject is how many modules hold a
    copy. That was a source test wearing an address count's name, and a fourth
    live `monkeypatch` target sat behind it with this file at 23 passed.
    `app.stt.routing` is where these names are defined; it is not where a copy
    has to come from.

    Counted by name and not by object, because a static walk has no way to
    tell `app.embeddings.clear_cache` from `app.stt.routing.clear_cache` at
    the import line. That direction is the safe one: an over-count is an entry
    in the allowlist below, which a reviewer reads in the diff, and an
    under-count is the defect.

    A plain `import app.stt.routing` binds the module and copies nothing, so
    every read through it lands on the attribute a `monkeypatch` replaces. A
    function-body import binds a local rather than a module attribute and is
    deliberately not counted; that is what `app/main.py`'s one deferral buys.
    A module-level `_lookup = routing.get_provider` is counted now and was the
    admitted blind spot. An address built by an expression this walk does not
    read -- a class attribute, a dict value, a default argument, `setattr` on
    the module object, a name computed at run time -- is still past it and is
    not claimed; `_unreadable_attribute_shapes` takes the two shapes that hide
    a whole namespace out of that list and makes them failures."""
    bound: set[str] = set()
    for node in _module_level_statements(_tree(path)):
        if isinstance(node, ast.ImportFrom):
            bound.update(alias.name for alias in node.names if alias.name in re_exported)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr in re_exported:
                bound.add(value.attr)
            elif isinstance(value, ast.Name) and value.id in re_exported:
                bound.add(value.id)
    return bound


def test_the_routing_names_gain_no_third_address_under_app():
    """A routing function reachable as `<module>.<name>` from anywhere but
    `app.stt.routing` is a second `monkeypatch` target, whether the import
    that made it named the package or the routing module.

    The gate above measures the import spelling and stayed green through the
    change that made this one necessary. Hoisting five function-body imports
    in `app/stt/local_setup.py` to a module-level `from app.stt.routing import
    get_local_load_error, get_provider, is_model_loaded, peek_local_provider`
    was the right spelling and still put those four functions at a third
    address: replacing `app.stt.routing.is_model_loaded` left `check_status()`
    reading the copy, and forty-six `monkeypatch` targets in
    `tests/test_local_setup.py` moved onto the copy to keep the suite green --
    the second-target defect the gate above exists to name, arriving through
    the spelling it endorses. That module binds the routing module now and
    calls `routing.is_model_loaded()`, so a patch at the canonical address
    lands.

    Two entries survive and both predate this pin, which is checkable rather
    than argued: on `2e2d099`, `app/stt/router.py:11` read `from app.stt
    import clear_cache, get_provider` and `app/pipeline/service.py:20` read
    `from app.stt import get_routed_provider, is_local_provider`. Spec 172
    changed which module those two name; it did not change how many copies
    they keep, and forty-three test sites patch
    `app.pipeline.service.get_routed_provider` at the address one of them
    makes. Adding a third entry is the claim that a module needs its own copy.

    `app/stt/routing.py` defines the names and `app/stt/__init__.py` is the
    re-export the package exists for, so neither is measured.

    What this covers, stated so the docstring claims no more than the walk
    proves: a module attribute bound by an import in any spelling, from any
    source module, and one bound by a plain module-level assignment off an
    attribute or a name. What it does not: an address built by an expression
    it does not read -- a class attribute, a dict value, a default argument,
    `setattr` on the module object, a name assembled at run time. Those are
    past a static walk, and the runtime method that closed the identity half
    of this file is not available here, because measuring every module's
    attributes means importing every module under `app/` and
    `app/audio/windows_loopback.py` imports `pyaudiowpatch`, which does not
    exist on macOS. The two shapes that would hide a whole namespace rather
    than one name are failures instead: `_unreadable_attribute_shapes`."""
    re_exported = _stt_routing_re_exports()
    assert re_exported, (
        "app/stt/__init__.py re-exports nothing from app.stt.routing, so this "
        "gate is matching an empty set and would pass against any diff. The "
        "routing module was renamed or the re-export block was removed; "
        "update _STT_ROUTING_MODULE."
    )

    measured = {}
    unreadable = {}
    for module, path in sorted(_modules().items()):
        if module in (_STT_PACKAGE, _STT_ROUTING_MODULE):
            continue
        relative = path.relative_to(APP_DIR.parent).as_posix()
        shapes = _unreadable_attribute_shapes(path)
        if shapes:
            unreadable[relative] = shapes
        bound = _routing_names_bound_at_module_level(
            path, _containing_package(path), re_exported
        )
        if bound:
            measured[relative] = sorted(bound)

    assert not unreadable, (
        f"these modules build module attributes this walk cannot enumerate: "
        f"{unreadable}. A star import binds whatever the other module holds "
        "and a module-level `__getattr__` serves names nothing binds, so "
        "either one can put a routing function at a third address with no "
        "statement here naming it and this gate counting zero. Spell the "
        "imports out, and reach a lazy module through `import <module>` "
        "instead of serving its names off this one."
    )

    assert measured == {
        relative: sorted(names) for relative, names in _MAY_HOLD_A_ROUTING_NAME.items()
    }, (
        f"pinned:   {_MAY_HOLD_A_ROUTING_NAME}; measured: {measured}. A "
        "module-level import of a routing name, or an assignment holding one, "
        "copies the function onto the importing module whatever module it was "
        "taken from, so `monkeypatch` has two addresses to "
        "choose from and a test that picks the other one asserts nothing. Bind "
        "the module instead - `from app.stt import routing`, then "
        "`routing.<name>()` - which reads the one attribute at call time. "
        "Adding an entry to _MAY_HOLD_A_ROUTING_NAME claims the copy is older "
        "than this pin, and the test sites patching it are the evidence."
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
    a cycle this returns nothing for. Three of this graph's four elementary
    cycles are of that kind. `app.main` is excluded because building the
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
    holds 4 entries here and 11 after one added function-local import, so it
    records a snapshot rather than a rule, while this membership moves by two
    readable names on that same edge.

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


def test_every_backend_package_is_classified_as_feature_boundary_or_shared():
    """The rule above only sees the names in `_FEATURE_PACKAGES` and
    `_HTTP_BOUNDARY_PACKAGES`, so a package absent from both is exempt in full
    rather than checked.

    The same hole `test_every_backend_package_is_covered_by_the_web_framework_allowlist`
    closes for the web-framework gate, and it is not hypothetical here:
    `app/llm/` existed until JS-169 deleted it, so packages do get added and
    removed. Without this, a new `app/newpkg/` could be imported from inside
    `core` and every rule in this file would stay green.

    Mutation-checked: creating a directory under `app/` that is named in none
    of the three sets reddens this test and nothing else.
    """
    classified = _FEATURE_PACKAGES | _HTTP_BOUNDARY_PACKAGES | _NON_FEATURE_PACKAGES
    unclassified = [
        package for package in _package_directories() if package not in classified
    ]

    assert not unclassified, (
        f"These packages are in no set, so the rule over core does not see "
        f"them: {unclassified}. Add each one to _FEATURE_PACKAGES, to "
        "_HTTP_BOUNDARY_PACKAGES if it holds routes, middleware or exception "
        "handlers, or to _NON_FEATURE_PACKAGES if it is shared ground every "
        "package may import."
    )


def test_shared_ground_is_pinned_and_the_three_classifications_do_not_overlap():
    """The coverage rule above reads a union, so it asks only whether a package
    is filed somewhere and cannot tell a correct filing from a wrong one.

    Moving a name between the sets changes which rule sees it while leaving
    every gate in this file green. Shared ground is pinned non-empty and
    against the tree rather than by name, so a second package filed there
    inherits every rule stated over the classification instead of forcing this
    assertion to be re-edited. The overlap check asserts first: a package filed
    twice is the more specific diagnosis, and the first `assert` to fire hides
    the second (ADR 082)."""
    classifications = {
        "_FEATURE_PACKAGES": _FEATURE_PACKAGES,
        "_HTTP_BOUNDARY_PACKAGES": _HTTP_BOUNDARY_PACKAGES,
        "_NON_FEATURE_PACKAGES": _NON_FEATURE_PACKAGES,
    }
    names = sorted(classifications)
    overlaps = [
        f"{left} and {right}: {sorted(classifications[left] & classifications[right])}"
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if classifications[left] & classifications[right]
    ]

    assert not overlaps, (
        f"These classification sets share a package: {overlaps}. A package "
        "filed twice is read as whichever set a given rule happens to consult, "
        "which is the hole the third set exists to close (ADR 082)."
    )
    assert _NON_FEATURE_PACKAGES, (
        "Shared ground is empty, so every rule stated over it checks nothing "
        "and the rule over core has no layer left to allow. See ADR 082."
    )
    unreal = [
        package
        for package in sorted(_NON_FEATURE_PACKAGES)
        if not (APP_DIR / package).is_dir()
    ]
    assert not unreal, (
        f"These packages are filed as shared ground but are no directory "
        f"under app/: {unreal}. Every name there is a layer app/core/ may "
        "import, so one that describes nothing widens the rule over core "
        "against a package that does not exist. See ADR 082."
    )


def test_the_mutually_dependent_package_set_has_not_changed():
    """Four packages all reach each other, so none can be read, moved or
    tested without the other three. This pins which four.

    Every such group is pinned, not the largest one. Returning only the biggest
    left a second tangle invisible and made the answer depend on dictionary
    order when two tied on size, so a three-package loop sharing no edge with
    the largest passed unseen and the test could change verdict between runs on
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
    ("app.transcripts.relocation", "app.transcripts.history", "_close_conn_locked"),
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
    name bound to an `app` module -- reads, calls and assignments alike, the
    assignment arm pinned by
    `test_the_package_private_walk_sees_a_write_into_another_modules_global` --
    and a direct `from app.x.y import _name`. Covering only the first would leave
    the rule one import line away from irrelevance, exactly as matching the
    literal `fastapi` did for rule 3.
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
    return _underscore_reach_ins_in(module, _tree(path), _containing_package(path), modules)


def _underscore_reach_ins_in(
    module: str, tree: ast.Module, package: str, modules: Mapping[str, Path]
) -> list[tuple[str, str, str]]:
    """The walk itself, over an already-parsed tree and the package it lives in.

    Separated from the file so a source string can be walked directly: the
    spellings this has to catch include ones no module under `app/` writes today.
    """
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
    one call of a sibling's package-private helper. The package writes no
    assignment into another module's global, so that spelling is pinned by
    `test_the_package_private_walk_sees_a_write_into_another_modules_global`
    instead."""
    missing = sorted(_LIVE_PACKAGE_PRIVATE_REACH_INS - _all_underscore_reach_ins())

    assert not missing, (
        f"The walk no longer finds these reach-ins: {missing}. Either the code "
        "moved -- repoint this set at reach-ins that are actually written -- or "
        "the walk stopped seeing a spelling it used to see, which makes the two "
        "tests above pass while checking nothing."
    )


_WRITE_REACH_IN_SOURCE = """
from app.transcripts import history


def point_the_store_somewhere(directory):
    history._output_dir = directory
"""


def test_the_package_private_walk_sees_a_write_into_another_modules_global():
    """The attribute arm has to catch an assignment, not only a read or a call.

    Every reach-in `app/` writes today is one of those two, so the live set
    above cannot tell this walk from one restricted to `ast.Load` -- both pass
    it. This source carries the spelling the package has none of, which is the
    one a module acquires on the day it starts writing a sibling's state."""
    found = _underscore_reach_ins_in(
        "app.transcripts.relocation",
        ast.parse(_WRITE_REACH_IN_SOURCE),
        "app.transcripts",
        _modules(),
    )

    assert ("app.transcripts.relocation", "app.transcripts.history", "_output_dir") in found, (
        f"The walk did not see an assignment into another module's global: {found}. "
        "A walk that only sees reads lets a module take ownership of a sibling's "
        "state without the gate above noticing (ADR 072)."
    )
