"""A backend route is live only if something names its path.

``[tool.vulture]`` in ``backend/pyproject.toml`` carries
``ignore_decorators = ["@router.*"]``, which is what makes ``min_confidence = 60``
affordable (ADR 051), and the exemption is structural: a FastAPI handler nobody
calls is exempt exactly like a live one. ``src/api-surface.test.ts`` closes only
the other half of the pair -- it reports an ``api`` member with no caller, never
a backend route with no client. This module closes the remaining half. ADR 054
records the design and its rejected alternatives.

**What it sees.** Every path in the OpenAPI schema ``app.main.app`` serves,
matched against the string and template literals scanned out of ``src/api.ts``.
Paths arrive from the schema with every router prefix already applied, which is
why the gate lives on the Python side: prefixes are declared both at
``app.include_router(...)`` and on ``APIRouter(prefix=...)``, so reading
decorator strings would yield the wrong path for six of the seven routers.

**Why the schema rather than ``app.routes``.** Walking ``app.routes`` reads a
private structure, and it moved: a FastAPI newer than the one installed here
stopped flattening ``include_router`` into that list and nests the children
inside an ``_IncludedRouter`` wrapper, so the walk found four routes on CI and
thirty locally. ``app.openapi()`` is the public, documented surface the
application itself serves and does not depend on how routes are stored. The
unpinned ``fastapi>=0.115.0`` requirement that let local and CI diverge is
filed as JS-141.

**What it cannot see, stated rather than implied.**

* **Verb-level deadness.** A client literal matches a route by path alone. Two
  application paths carry two verbs each (``/history`` GET and DELETE,
  ``/settings`` GET and PUT); for those, one verb losing its caller is
  invisible. ``src/api.ts`` puts the verb in three different positions relative
  to the path, so every pairing rule covering all three is positional and
  approximate.
* **A static literal segment standing in a parameter position.** ``/history/staats``
  matches ``/history/{entry_id}`` and keeps it alive. That is not a defect to
  be fixed: such a request is structurally valid and the route really would
  answer it, so nothing here can know the segment was a typo rather than an
  identifier. An interpolated segment is the opposite case and is *not*
  permitted to stand in for a static one -- see ``paths_match``.
* **A path the client never spells as a literal.** One assembled by
  concatenation makes its route look dead -- a loud failure toward an exemption
  rather than a silent pass.
* **A path literal held by an unexported, uncalled helper inside
  ``src/api.ts``.** knip does not report unexported symbols and
  ``tsconfig.json`` sets ``strict`` without ``noUnusedLocals``, so nothing in
  the repository would report such a helper.
* **A client reaching the backend by a mechanism outside
  ``CLIENT_REQUEST_OPENERS``.** That tuple is what
  ``test_api_ts_is_the_whole_client_surface`` looks for, and a transport nobody
  listed there -- a future ``navigator.sendBeacon``, say -- would let a second
  client surface grow outside ``src/api.ts`` unseen, and every route it alone
  consumes would be reported dead.
* **Two literals collapsing to one path.** Whether a route is consumed is
  decided per literal, and the caller-less members of
  ``CLIENT_MEMBERS_WITHOUT_CALLER`` are excluded by the member that spells
  them, not by the path they name. A dead member and a live one may therefore
  address the same route, and the live one keeps it alive -- which is the
  point. What stays invisible is the reverse: a route whose only live literal
  is a coincidence of query-string trimming.

**Why every file it reads is named explicitly.** ``backend/build/`` and
``backend/.venv-build/Lib/site-packages/app/`` hold stale full copies of
``app/`` on any machine that has run an editable install, so a glob from the
repository root would silently measure the wrong tree. Two globs are used and
both stay inside a tree that holds no build output: ``backend/app/**/*.py``,
which is inside the package itself, and ``src/**/*.ts``, which is the
TypeScript source tree and neither a build target nor a dependency root. This
is the same constraint ``test_cross_language_contracts.py`` records.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from typing import NamedTuple

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]

API_TS = REPO_ROOT / "src" / "api.ts"
API_SURFACE_TEST_TS = REPO_ROOT / "src" / "api-surface.test.ts"
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"
SRC_DIR = REPO_ROOT / "src"
APP_DIR = REPO_ROOT / "backend" / "app"

INTERPOLATION_WILDCARD = "\x00interpolation\x00"

HTTP_VERB_DECORATORS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})

CONSUMED_OUTSIDE_TYPESCRIPT: dict[str, tuple[tuple[Path, str], ...]] = {
    "/shutdown": ((BACKEND_RS, r'format!\("http://127\.0\.0\.1:\{\}/shutdown", PORT\)'),),
}
"""Routes kept alive by a caller outside ``src/``, each site named with the
regular expression that file must still match.

A regular expression rather than a substring search: ``/shutdown`` also appears
in doc comments in the same Rust file that call nothing, so a substring search
would pass on a file whose only real call site had been deleted. This is the
declared-sites shape ``test_cross_language_contracts.py`` uses.

``/health`` needs no entry despite its Rust and Python consumers
(``src-tauri/src/backend.rs``, ``backend/scripts/smoke_sidecar.py``), because
the ``health`` member of ``src/api.ts`` already calls it -- and an entry here
whose route turns out to have a client literal after all is reported as
redundant below.
"""

UNCONSUMED_PENDING_A_DECISION: dict[str, str] = {
    "/audio/stop": (
        "JS-122 moved the microphone test to /audio/discard. This route is still the only "
        "way to end a capture and keep its WAV, and that ownership contract is pinned by "
        "backend/tests/test_audio.py and backend/tests/test_meeting_recorder.py. Deleting "
        "it is an API-surface decision against the session contract of ADR 050, not a "
        "dead-code cleanup."
    ),
    "/stt/local/install": (
        "Local-engine packaging is blocked on the Free/Pro product split, which is the "
        "user's decision. Deleting the route would pre-empt it silently."
    ),
    "/stt/local/load": (
        "Listed as a caller-less api member in src/api-surface.test.ts for the reason "
        "recorded there: removing only the client half leaves a wired-up backend half "
        "with nothing driving it, and removing both halves is an API-surface decision "
        "nobody has taken."
    ),
    "/stt/local/unload": (
        "Same decision as /stt/local/load, and it moves with it."
    ),
}
"""Wired-up routes reached by nobody, each mapped to the decision that would
resolve it.

This is a live list, not a suppression. An entry fails when its route stops
existing, and it fails again when its route acquires a client consumer, so it
cannot quietly become the place dead endpoints go to be forgiven.
"""

CLIENT_MEMBERS_WITHOUT_CALLER: dict[str, str] = {
    "sttLocalLoad": "/stt/local/load",
    "sttLocalUnload": "/stt/local/unload",
}
"""``api`` members that ``src/api-surface.test.ts`` already calls dead, mapped
to the route each one names.

Both members spell their path in ``src/api.ts``, so a literal scan alone would
certify their routes as live while the neighbouring gate calls their only
callers dead. The exclusion is by member rather than by path: a literal is
dropped because of where it sits, so a second, genuinely called member spelling
the same path still consumes the route and the entry in
``UNCONSUMED_PENDING_A_DECISION`` fires as it advertises. The key set is
asserted equal to the array read out of ``src/api-surface.test.ts`` so the two
gates cannot drift apart.
"""

CLIENT_REQUEST_OPENERS = ("fetch(", "EventSource(", "WebSocket(", "XMLHttpRequest(")
"""Every way a module in ``src/`` can open a request to the backend.

``fetch(`` alone would miss ``src/api.ts``'s own level-stream client if it were
ever rewritten onto ``EventSource``, which is the natural transport for the
``text/event-stream`` that route serves, and any module that grew such a client
outside ``src/api.ts`` would then be invisible to the scope check below.
"""

BACKEND_ADDRESS_TOKENS = ("BACKEND_BASE_URL", "127.0.0.1", "localhost")

NON_PRODUCTION_TYPESCRIPT_SUFFIXES = (".test.ts", ".test-helper.ts", ".d.ts")
"""Suffixes that mark a TypeScript module as something other than production
code.

``src/api-surface.test.ts`` names only ``.test.ts``, which makes
``src/settings/history-page-stub.test-helper.ts`` and any future ``*.d.ts``
production modules by its rule. This tuple is a superset of that rule, and
``test_the_production_module_rule_covers_the_typescript_gate_s_own`` pins the
relationship so the two gates cannot end up disagreeing about which files count.
"""

TYPESCRIPT_ALLOWLIST = re.compile(
    r"const\s+ALLOWED_WITHOUT_PRODUCTION_CALLER\s*=\s*\[(.*?)\]",
    re.DOTALL,
)

TYPESCRIPT_PRODUCTION_RULE = re.compile(
    r"function\s+productionSources\b.*?"
    r"!entry\.endsWith\(\"\.ts\"\)"
    r"((?:\s*\|\|\s*entry\.endsWith\(\"[^\"]+\"\))+)",
    re.DOTALL,
)

API_MEMBER_KEY = re.compile(r"^  ([A-Za-z_$][\w$]*)\s*:", re.MULTILINE)


class RouterDecorator(NamedTuple):
    """One ``@router.<attribute>(...)`` decorator found under ``backend/app``."""

    module: str
    attribute: str
    keywords: tuple[str, ...]


class ScannedSource(NamedTuple):
    """A TypeScript source read once: its literals, its code, and where it ended."""

    literals: tuple[tuple[int, str], ...]
    code_only: str
    final_mode: str


@functools.cache
def application_schema_paths() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Every path in the served OpenAPI schema, with the verbs declared on it.

    ``app.openapi()`` is FastAPI's public surface and applies every router
    prefix itself, so this needs no knowledge of how the application wires its
    routers together. A path item may also carry non-operation keys
    (``parameters``, ``summary``, ``$ref``), which is why the verbs are filtered
    against ``HTTP_VERB_DECORATORS`` rather than taken as the whole key set.
    """
    schema = app.openapi()
    return tuple(
        (path, tuple(sorted(key.upper() for key in item if key in HTTP_VERB_DECORATORS)))
        for path, item in sorted(schema.get("paths", {}).items())
    )


@functools.cache
def application_paths() -> frozenset[str]:
    """The application's route paths."""
    return frozenset(path for path, _ in application_schema_paths())


@functools.cache
def router_decorators() -> tuple[RouterDecorator, ...]:
    """Every ``@router.<attribute>(...)`` decorator declared under ``backend/app``.

    Read with ``ast`` rather than a text pattern: a decorator spelled inside a
    docstring is not a node, and the attribute name is available exactly rather
    than as whatever a regular expression was told to look for -- which is how
    ``@router.api_route`` becomes visible instead of silently uncounted.
    """
    found: list[RouterDecorator] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                call = decorator if isinstance(decorator, ast.Call) else None
                target = call.func if call is not None else decorator
                if not isinstance(target, ast.Attribute):
                    continue
                if not isinstance(target.value, ast.Name) or target.value.id != "router":
                    continue
                keywords = tuple(
                    keyword.arg
                    for keyword in (call.keywords if call is not None else [])
                    if keyword.arg is not None
                )
                found.append(
                    RouterDecorator(
                        module=path.relative_to(REPO_ROOT).as_posix(),
                        attribute=target.attr,
                        keywords=keywords,
                    )
                )
    return tuple(found)


def _starts_a_regular_expression(previous: str) -> bool:
    """Whether a ``/`` following ``previous`` opens a regex rather than divides.

    The full JavaScript rule needs the parser's state; this is the practical
    half of it -- a regex can only begin where a value can, so it is division
    exactly when the last thing before it could end one.
    """
    return previous == "" or previous in "([{,;:=!&|?+-*%^~<>"


def scan_source(source: str) -> ScannedSource:
    """Every string, single-quoted and template literal in ``source`` with its
    offset, the source with comments and literal contents blanked out, and the
    mode the scanner ended in.

    Comments and regular-expression literals are skipped *before* quotes are
    honoured. ``src/api.ts`` carries prose apostrophes inside ``/** ... */``
    blocks (``Tauri's``, ``ADR 049's``), and a character class such as
    ``/['"]/g`` carries them in code; a scanner that enters quote mode on either
    desynchronises and silently loses literals -- and with an even number of
    quote characters it resynchronises further down, so the file still reads to
    its end and the loss leaves no trace.

    A template literal is read whole, interpolations included: its contents are
    matched as a pattern later rather than parsed here, so a ``{``, ``)`` or
    ``;`` inside one is data. An escaped character inside any literal is
    consumed as a pair, so ``\\"`` does not end the literal.

    The blanked source preserves offsets and line structure, so a pattern run
    over it -- the ``api`` member keys, here -- cannot match inside a comment or
    a string. It is the shape ``blankCommentsAndStrings`` in
    ``src/api-surface.test.ts`` uses for the same reason.

    The returned mode is the precise statement of "the scanner read the whole
    file rather than a prefix": anything but ``code`` means an unterminated
    literal or comment swallowed the tail.
    """
    literals: list[tuple[int, str]] = []
    blanked: list[str] = []
    current: list[str] = []
    start = 0
    mode = "code"
    previous_code_char = ""
    index = 0

    def blank(count: int) -> None:
        for offset in range(count):
            char = source[index + offset]
            blanked.append("\n" if char == "\n" else " ")

    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if mode == "code":
            if char == "/" and following == "/":
                mode = "line-comment"
                blank(2)
                index += 2
            elif char == "/" and following == "*":
                mode = "block-comment"
                blank(2)
                index += 2
            elif char == "/" and _starts_a_regular_expression(previous_code_char):
                mode = "regex"
                blank(1)
                index += 1
            elif char in ("'", '"', "`"):
                mode = char
                current = []
                start = index
                blank(1)
                index += 1
            else:
                blanked.append(char)
                if not char.isspace():
                    previous_code_char = char
                index += 1
            continue

        if mode == "line-comment":
            if char == "\n":
                mode = "code"
            blank(1)
            index += 1
            continue

        if mode == "block-comment":
            if char == "*" and following == "/":
                mode = "code"
                blank(2)
                index += 2
            else:
                blank(1)
                index += 1
            continue

        if mode == "regex":
            if char == "\\":
                blank(2)
                index += 2
                continue
            if char == "/":
                mode = "code"
                previous_code_char = "/"
            blank(1)
            index += 1
            continue

        if char == "\\":
            current.append(source[index : index + 2])
            blank(2)
            index += 2
            continue
        if char == mode:
            literals.append((start, "".join(current)))
            mode = "code"
            previous_code_char = char
            blank(1)
            index += 1
            continue
        current.append(char)
        blank(1)
        index += 1

    return ScannedSource(tuple(literals), "".join(blanked), mode)


def _is_backend_path(value: str) -> bool:
    """Whether a literal looks like a backend path.

    A leading ``/`` and not ``//``: relative imports start with ``.`` and URLs
    with ``h``, so the filter yields no false candidates on ``src/api.ts``.
    """
    return value.startswith("/") and not value.startswith("//")


def _owning_member(members: tuple[tuple[int, str], ...], offset: int) -> str | None:
    """The ``api`` member a literal at ``offset`` belongs to, if any."""
    owner: str | None = None
    for start, name in members:
        if start > offset:
            break
        owner = name
    return owner


def client_path_literals(source: str, dead_members: frozenset[str] = frozenset()) -> set[str]:
    """The literals from ``source`` that look like a backend path, excluding
    those spelled inside a member of ``dead_members``.

    A literal is attributed to the nearest preceding two-space-indented object
    key in the blanked source, which is how every member of ``api`` is written.
    The attribution is what makes the exclusion per literal rather than per
    path: a route named by both a caller-less member and a live one stays
    consumed, so ``UNCONSUMED_PENDING_A_DECISION`` still fires when its route
    acquires a real caller.
    """
    scanned = scan_source(source)
    members = tuple(
        (match.start(), match.group(1)) for match in API_MEMBER_KEY.finditer(scanned.code_only)
    )
    return {
        value
        for offset, value in scanned.literals
        if _is_backend_path(value) and _owning_member(members, offset) not in dead_members
    }


@functools.cache
def api_ts_source() -> str:
    """``src/api.ts``, read once."""
    return API_TS.read_text(encoding="utf-8")


def normalise_path(path: str) -> tuple[str, ...]:
    """``path`` as comparable segments.

    Each ``${...}`` interpolation collapses to a wildcard token first, so a
    ``?`` appearing inside an interpolated expression cannot be mistaken for the
    start of the query string; then everything from the first ``?`` is dropped.
    """
    collapsed = re.sub(r"\$\{.*?\}", INTERPOLATION_WILDCARD, path)
    return tuple(collapsed.split("?", 1)[0].split("/"))


def _is_parameter(route_segment: str) -> bool:
    """Whether a route segment is a FastAPI ``{param}`` placeholder."""
    return route_segment.startswith("{") and route_segment.endswith("}")


def paths_match(route_path: str, literal: str) -> bool:
    """Whether ``literal`` addresses ``route_path``.

    Segment by segment over lists of equal length. A FastAPI ``{param}``
    placeholder matches any literal segment, and a literal segment carrying an
    interpolation matches a ``{param}`` placeholder and nothing else.

    That second rule is the whole of the gate's soundness. A segment containing
    a value the client computes at run time cannot be claimed to address a
    static route: letting ``/history/${id}`` stand for any ``/history/<x>``
    certified ``/history/search``, ``/history/stats`` and every future sibling
    as consumed on the strength of one unrelated literal -- a silent false
    negative in the one direction this gate exists to catch. The reverse
    direction stays permissive on purpose and is stated in the module docstring:
    a static literal in a parameter position really does reach that route.
    """
    route_segments = normalise_path(route_path)
    literal_segments = normalise_path(literal)
    if len(route_segments) != len(literal_segments):
        return False
    return all(
        _is_parameter(route_segment)
        or (INTERPOLATION_WILDCARD not in literal_segment and route_segment == literal_segment)
        for route_segment, literal_segment in zip(route_segments, literal_segments)
    )


def routes_addressed_by(literal: str, paths: frozenset[str]) -> set[str]:
    """The routes in ``paths`` that ``literal`` consumes.

    An exact match wins outright. ``/history/search`` is a route in its own
    right and also a structurally valid request to ``/history/{entry_id}``;
    without this rule the literal that names the first would keep the second
    alive as well, which is the parameter-position permissiveness leaking into
    routes that have their own consumer question to answer.
    """
    literal_segments = normalise_path(literal)
    exact = {path for path in paths if normalise_path(path) == literal_segments}
    if exact:
        return exact
    return {path for path in paths if paths_match(path, literal)}


@functools.cache
def consumed_routes() -> frozenset[str]:
    """Route paths a live client literal addresses."""
    paths = application_paths()
    literals = client_path_literals(api_ts_source(), frozenset(CLIENT_MEMBERS_WITHOUT_CALLER))
    consumed: set[str] = set()
    for literal in literals:
        consumed |= routes_addressed_by(literal, paths)
    return frozenset(consumed)


@functools.cache
def typescript_allowlist_members() -> tuple[str, ...]:
    """``ALLOWED_WITHOUT_PRODUCTION_CALLER`` as declared in
    ``src/api-surface.test.ts``.

    Read with a regular expression that tolerates the array being reformatted
    across lines, since prettier splitting it is a formatting change rather than
    a contract change.
    """
    source = API_SURFACE_TEST_TS.read_text(encoding="utf-8")
    match = TYPESCRIPT_ALLOWLIST.search(source)
    assert match is not None, (
        f"no `const ALLOWED_WITHOUT_PRODUCTION_CALLER = [...]` declaration was found in "
        f"{API_SURFACE_TEST_TS.name}. It was renamed or removed, and the mirror below "
        f"cannot check two gates agree while it cannot read one of them"
    )
    return tuple(re.findall(r'"([^"]*)"', match.group(1)))


@functools.cache
def production_typescript_modules() -> tuple[tuple[Path, str], ...]:
    """Every production TypeScript module under ``src/``, by path."""
    return tuple(
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(SRC_DIR.rglob("*.ts"))
        if not path.name.endswith(NON_PRODUCTION_TYPESCRIPT_SUFFIXES)
    )


def _addresses_the_backend(source: str) -> bool:
    """Whether a module opens a request to the backend at all, however it names
    the host and whatever transport it uses.

    ``BACKEND_BASE_URL`` alone would miss a module that hardcodes the host
    instead of importing the constant -- the seam ``docs/style-guide.md`` already
    names -- and that module's routes would be reported dead by this file.
    """
    return any(opener in source for opener in CLIENT_REQUEST_OPENERS) and any(
        token in source for token in BACKEND_ADDRESS_TOKENS
    )


def test_the_scanner_reads_api_ts_to_its_end():
    scanned = scan_source(api_ts_source())

    assert scanned.final_mode == "code", (
        f"the literal scanner ended {API_TS.name} in `{scanned.final_mode}` mode, so it "
        f"read a prefix of the file and stopped. Every path literal after the cut is "
        f"missing and every assertion in this module is silently incomplete"
    )
    assert scanned.literals, f"the scanner found no string literals at all in {API_TS.name}"


def test_every_application_route_has_a_consumer():
    paths = application_paths()

    assert paths, (
        "app.main.app serves an OpenAPI schema with no paths at all, so every assertion "
        "in this module is trivially true over an empty set. It is the gate that is "
        "broken, not the routes. The schema holds "
        f"{sorted(app.openapi().keys())} at the top level, app.routes holds "
        f"{len(app.routes)} entries of types {sorted({type(r).__name__ for r in app.routes})}, "
        f"and app.main was imported from "
        f"{getattr(__import__('app.main', fromlist=['__file__']), '__file__', '<unknown>')}"
    )

    unconsumed = (
        set(paths)
        - consumed_routes()
        - set(CONSUMED_OUTSIDE_TYPESCRIPT)
        - set(UNCONSUMED_PENDING_A_DECISION)
    )

    assert unconsumed == set(), (
        f"these backend routes are addressed by no path literal in src/api.ts: "
        f"{sorted(unconsumed)}. Either delete the route and its client half together "
        f"(the handler, its response model if nothing else uses it, and its test), or add "
        f"an entry with its reason: CONSUMED_OUTSIDE_TYPESCRIPT if a caller outside src/ "
        f"keeps it alive, UNCONSUMED_PENDING_A_DECISION if it is waiting on a product "
        f"decision (ADR 054)"
    )


def test_every_client_literal_addresses_a_route():
    paths = application_paths()
    orphans = sorted(
        literal
        for literal in client_path_literals(api_ts_source())
        if not routes_addressed_by(literal, paths)
    )

    assert orphans == [], (
        f"these path literals in src/api.ts address no backend route: {orphans}. The "
        f"client is pointing at a renamed or deleted endpoint and would fail at runtime"
    )


def test_the_route_set_matches_the_router_decorator_sites():
    decorator_sites = sum(
        1 for decorator in router_decorators() if decorator.attribute in HTTP_VERB_DECORATORS
    )
    registered_verbs = sum(len(verbs) for _, verbs in application_schema_paths())

    assert decorator_sites == registered_verbs, (
        f"{decorator_sites} `@router.<verb>` decorator sites under backend/app do not "
        f"account for the {registered_verbs} verbs app.main.app publishes. A route "
        f"reached this application by some other mechanism -- registered directly on the "
        f"app object, "
        f"or added with app.add_api_route -- and this module's rule has to be extended to "
        f"see it before it can claim to check every route"
    )


def test_no_router_decorator_registers_verbs_this_module_cannot_count():
    uncountable = sorted(
        f"{decorator.module}: @router.{decorator.attribute}"
        for decorator in router_decorators()
        if decorator.attribute not in HTTP_VERB_DECORATORS
    )

    assert uncountable == [], (
        f"these decorators register routes without being one verb each: {uncountable}. "
        f"`@router.api_route(methods=[...])` is the shape that does it -- it adds as many "
        f"verbs as its list holds while the count above sees a single site, so the "
        f"decorator cross-check would read a mismatch it cannot explain"
    )


def test_no_route_hides_itself_from_the_schema():
    hidden = sorted(
        f"{decorator.module}: @router.{decorator.attribute}"
        for decorator in router_decorators()
        if "include_in_schema" in decorator.keywords
    )

    assert hidden == [], (
        f"these routes pass include_in_schema to their decorator: {hidden}. This module "
        f"enumerates routes from the served OpenAPI schema, so a route excluded from it "
        f"is invisible here and would be exempt from the consumer check without anything "
        f"saying so"
    )


def test_no_declared_outside_consumer_has_outlived_its_route():
    stale = sorted(path for path in CONSUMED_OUTSIDE_TYPESCRIPT if path not in application_paths())

    assert stale == [], (
        f"these CONSUMED_OUTSIDE_TYPESCRIPT entries name routes that no longer exist: "
        f"{stale}. A stale entry silently exempts whatever future route reuses the path"
    )


def test_every_declared_outside_consumer_site_still_calls_its_route():
    broken = []
    for route_path, sites in CONSUMED_OUTSIDE_TYPESCRIPT.items():
        for site, pattern in sites:
            if not re.search(pattern, site.read_text(encoding="utf-8")):
                broken.append(f"{route_path} at {site.relative_to(REPO_ROOT).as_posix()}")

    assert broken == [], (
        f"these declared call sites no longer match the pattern that proves they call "
        f"their route: {broken}. Either the call moved and the pattern needs updating, or "
        f"the caller is gone and the route is now unconsumed"
    )


def test_no_declared_outside_consumer_is_redundant():
    redundant = sorted(set(CONSUMED_OUTSIDE_TYPESCRIPT) & consumed_routes())

    assert redundant == [], (
        f"these routes have a client literal in src/api.ts, so their "
        f"CONSUMED_OUTSIDE_TYPESCRIPT entries carry no weight and only add a second place "
        f"to keep correct: {redundant}"
    )


def test_no_pending_decision_entry_has_outlived_its_route():
    stale = sorted(
        path for path in UNCONSUMED_PENDING_A_DECISION if path not in application_paths()
    )

    assert stale == [], (
        f"these UNCONSUMED_PENDING_A_DECISION entries name routes that no longer exist: "
        f"{stale}. The decision was taken and the route deleted; delete its entry too, or "
        f"a future route reusing the path inherits the exemption"
    )


def test_no_pending_decision_entry_has_acquired_a_consumer():
    resolved = sorted(set(UNCONSUMED_PENDING_A_DECISION) & consumed_routes())

    assert resolved == [], (
        f"these routes now have a consumer, so they are no longer pending anything and "
        f"their UNCONSUMED_PENDING_A_DECISION entries must be deleted: {resolved}. The "
        f"list is a record of visible debt, not a permanent exemption"
    )


def test_the_caller_less_client_members_mirror_the_typescript_allowlist():
    declared = set(CLIENT_MEMBERS_WITHOUT_CALLER)
    typescript = set(typescript_allowlist_members())

    assert declared == typescript, (
        f"CLIENT_MEMBERS_WITHOUT_CALLER and ALLOWED_WITHOUT_PRODUCTION_CALLER in "
        f"src/api-surface.test.ts no longer agree: this module has {sorted(declared)}, "
        f"that file has {sorted(typescript)}. The two gates would then disagree about "
        f"which client members are dead, and a route whose only literal sits in a dead "
        f"member would be certified live by one of them"
    )


def test_every_caller_less_client_member_names_a_real_route():
    paths = application_paths()
    stale = sorted(
        f"{member} -> {path}"
        for member, path in CLIENT_MEMBERS_WITHOUT_CALLER.items()
        if path not in paths
    )

    assert stale == [], (
        f"these CLIENT_MEMBERS_WITHOUT_CALLER entries name routes that no longer exist: "
        f"{stale}. The mapping excludes these members' literals from the consumed set, so "
        f"a stale one excludes nothing and hides its member's real route"
    )


def test_every_caller_less_client_member_is_found_in_api_ts():
    scanned = scan_source(api_ts_source())
    declared = {match.group(1) for match in API_MEMBER_KEY.finditer(scanned.code_only)}
    missing = sorted(set(CLIENT_MEMBERS_WITHOUT_CALLER) - declared)

    assert missing == [], (
        f"these CLIENT_MEMBERS_WITHOUT_CALLER members are not spelled as an object key in "
        f"src/api.ts: {missing}. Their literals are excluded by the member that owns "
        f"them, so a member this file cannot locate excludes nothing and its route is "
        f"certified live by a caller the neighbouring gate reports as dead"
    )


def test_the_production_module_rule_covers_the_typescript_gate_s_own():
    source = API_SURFACE_TEST_TS.read_text(encoding="utf-8")
    match = TYPESCRIPT_PRODUCTION_RULE.search(source)

    assert match is not None, (
        f"no `!entry.endsWith(\".ts\") || entry.endsWith(...)` rule was found inside "
        f"productionSources() in {API_SURFACE_TEST_TS.name}. That function is this "
        f"repository's definition of a production module and this one cannot be checked "
        f"against it unread"
    )
    excluded = re.findall(r'entry\.endsWith\("([^"]+)"\)', match.group(1))
    unmirrored = sorted(set(excluded) - set(NON_PRODUCTION_TYPESCRIPT_SUFFIXES))

    assert unmirrored == [], (
        f"src/api-surface.test.ts excludes {unmirrored} from its production modules and "
        f"NON_PRODUCTION_TYPESCRIPT_SUFFIXES does not, so this gate would scan a file that "
        f"one calls a test. The two must not disagree about the same file set"
    )


def test_api_ts_is_the_whole_client_surface():
    others = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path, source in production_typescript_modules()
        if path != API_TS and _addresses_the_backend(source)
    )

    assert others == [], (
        f"these production modules call the backend without going through src/api.ts: "
        f"{others}. This module scans src/api.ts alone, and that is only sound while "
        f"src/api.ts is the whole client surface -- a route consumed from one of these "
        f"files would be reported dead"
    )


SYNTHETIC_LINE_COMMENT = "\n".join(
    [
        "// Tauri's own poller hits /commented-only every 5 s.",
        'const path = "/plain";',
    ]
)
"""One apostrophe, in a ``//`` comment, and one real literal after it.

The apostrophe is the only ``'`` in the source, so a scanner that honours
quotes before it skips comments opens a single-quoted literal it never closes:
the assertions below then see no literal at all and a final mode that is not
``code``. Pairing it with a second apostrophe would let the two cancel out and
the mutation would pass silently, which is the failure this file exists to
avoid.
"""

SYNTHETIC_BLOCK_COMMENT = "\n".join(
    [
        "/** ADR 049's note: /also-commented-only is not a call site. */",
        'const path = "/plain";',
    ]
)
"""The same shape for a ``/** ... */`` block, and separate from the line-comment
case on purpose.

Held in one source, whichever comment came first would supply an apostrophe for
the other to close against, and the tail would resynchronise -- so disabling
either branch would leave every assertion green.
"""

SYNTHETIC_REGEX_LITERAL = "\n".join(
    [
        "const CONTROL = /['\"]/g;",
        'const path = "/plain";',
        "const TRAILING = /\"'/;",
    ]
)
"""A character class holding both quote characters, a real literal after it, and
a second regex closing the apostrophe count.

Two apostrophes rather than one on purpose: a scanner with no regex state opens
a single-quoted literal on the first and closes it on the second, so it ends in
``code`` mode having read the whole file, and ``final_mode`` reports nothing
wrong. The only visible damage is that ``/plain`` was swallowed as literal
content, which is exactly the silent loss this case pins.
"""

SYNTHETIC_LITERAL_SHAPES = "\n".join(
    [
        "const paths = {",
        '  plain: "/plain",',
        "  single: '/single',",
        "  interpolated: `/history/${entry.id}`,",
        "  query: `/words/top?lang=${lang}&limit=${limit}`,",
        "  awkward: `/awkward/{braced};and)parens`,",
        '  escaped: "/escaped\\"quote",',
        "  relative: '../not-a-path',",
        "  url: 'https://example.test/nope',",
        "};",
    ]
)
"""Every literal shape ``src/api.ts`` uses, plus the two shapes the path filter
must reject."""

SYNTHETIC_COMMENTED_PATHS = "\n".join(
    [
        '// The poller used to hit "/commented-only" before JS-122.',
        "/** And `/also-commented-only` was its meeting twin. */",
        'const path = "/plain";',
    ]
)
"""Paths spelled as literals inside comments, and one real literal.

No apostrophe here: the two sources above use one to prove the scanner stays
synchronised, and an unclosed quote would swallow these paths before the filter
ever saw them -- leaving the assertion green for the wrong reason.
"""

SYNTHETIC_DEAD_MEMBER_ONLY = "\n".join(
    [
        "export const api = {",
        '  sttLocalLoad: () => request("POST", "/stt/local/load", undefined, UNRECONCILED),',
        "};",
    ]
)
"""The route named only by the member ``src/api-surface.test.ts`` calls dead."""

SYNTHETIC_DEAD_MEMBER_AND_A_LIVE_ONE = "\n".join(
    [
        "export const api = {",
        '  sttLocalLoad: () => request("POST", "/stt/local/load", undefined, UNRECONCILED),',
        "",
        "  loadTheEngineFromTheWidget: () =>",
        '    request("POST", "/stt/local/load", undefined, UNRECONCILED),',
        "};",
    ]
)
"""The same route, named a second time by a member with a real caller.

This is the shape ``UNCONSUMED_PENDING_A_DECISION`` advertises it will catch:
the entry must stop being pending anything the moment a live client addresses
its route, and it cannot do that while the exclusion is applied to the path
rather than to the member that spells it.
"""


def test_a_prose_apostrophe_in_a_line_comment_does_not_desynchronise_the_scanner():
    scanned = scan_source(SYNTHETIC_LINE_COMMENT)

    assert scanned.final_mode == "code", (
        f"an apostrophe inside a `//` comment left the scanner in `{scanned.final_mode}` "
        f"mode. On src/api.ts that means silently losing every path literal after the "
        f"first piece of prose that contains one"
    )
    assert [value for _, value in scanned.literals] == ["/plain"]


def test_a_prose_apostrophe_in_a_block_comment_does_not_desynchronise_the_scanner():
    scanned = scan_source(SYNTHETIC_BLOCK_COMMENT)

    assert scanned.final_mode == "code", (
        f"an apostrophe inside a `/** */` block left the scanner in `{scanned.final_mode}` "
        f"mode. src/api.ts carries `Tauri's` and `ADR 049's` in exactly such blocks"
    )
    assert [value for _, value in scanned.literals] == ["/plain"]


def test_a_quote_inside_a_regex_literal_does_not_desynchronise_the_scanner():
    scanned = scan_source(SYNTHETIC_REGEX_LITERAL)

    assert [value for _, value in scanned.literals] == ["/plain"], (
        f"a quote character inside a regular-expression literal was honoured as the start "
        f"of a string, and the scanner read {[value for _, value in scanned.literals]} "
        f"instead. With an even number of such characters it resynchronises further down, "
        f"so the file still reads to its end and every literal in between is lost with "
        f"nothing reporting it"
    )
    assert scanned.final_mode == "code"


def test_a_path_named_only_in_a_comment_is_not_a_consumer():
    assert client_path_literals(SYNTHETIC_COMMENTED_PATHS) == {"/plain"}, (
        "a path quoted inside a comment was collected as a client literal. Such a path "
        "calls nothing, and counting it would certify a dead route as live on the "
        "strength of prose"
    )


def test_the_scanner_reads_every_literal_shape_whole():
    assert client_path_literals(SYNTHETIC_LITERAL_SHAPES) == {
        "/plain",
        "/single",
        "/history/${entry.id}",
        "/words/top?lang=${lang}&limit=${limit}",
        "/awkward/{braced};and)parens",
        '/escaped\\"quote',
    }


def test_a_literal_inside_a_caller_less_member_is_not_a_consumer():
    assert client_path_literals(
        SYNTHETIC_DEAD_MEMBER_ONLY, frozenset(CLIENT_MEMBERS_WITHOUT_CALLER)
    ) == set(), (
        "the literal spelled inside a member src/api-surface.test.ts reports as having no "
        "production caller was counted as a consumer, which would certify its route live "
        "while the only code naming it is dead"
    )


def test_a_second_live_member_revives_a_route_a_caller_less_one_names():
    assert client_path_literals(
        SYNTHETIC_DEAD_MEMBER_AND_A_LIVE_ONE, frozenset(CLIENT_MEMBERS_WITHOUT_CALLER)
    ) == {"/stt/local/load"}, (
        "a route named by both a caller-less member and a live one was still treated as "
        "unconsumed. UNCONSUMED_PENDING_A_DECISION advertises that it fails when its route "
        "acquires a consumer, and it cannot do that while the exclusion is by path"
    )


def test_an_interpolated_segment_matches_a_route_parameter():
    assert paths_match("/history/{entry_id}", "/history/${entry.id}")
    assert not paths_match("/history/{entry_id}", "/history/${entry.id}/extra")


def test_an_interpolated_segment_does_not_certify_a_static_sibling_route():
    siblings = frozenset({"/history/{entry_id}", "/history/search", "/history/canary-dead"})

    assert routes_addressed_by("/history/${entry.id}", siblings) == {"/history/{entry_id}"}, (
        "an interpolated segment matched static sibling routes. One literal spelling "
        "`/history/${id}` then certifies /history/search, /history/stats and every route "
        "added under /history as consumed, which is the silent false negative this gate "
        "exists to catch"
    )
    assert not paths_match("/audio/level-stream", "/audio/${kind}-stream")


def test_a_route_parameter_matches_a_plain_literal_segment():
    assert paths_match("/history/{entry_id}", "/history/42")


def test_an_exact_literal_consumes_only_the_route_it_names():
    routes = frozenset({"/history/search", "/history/{entry_id}"})

    assert routes_addressed_by("/history/search", routes) == {"/history/search"}, (
        "a literal naming a static route also kept its parameterised sibling alive. Every "
        "static route under /history would then consume /history/{entry_id}, and that "
        "route's own consumer question could never be asked"
    )


def test_a_query_string_is_dropped_at_the_first_question_mark():
    assert paths_match("/words/top", "/words/top?lang=${lang}&limit=${limit}")
    assert paths_match("/history", "/history?limit=${limit}&before_ts=${cursor.ts}")


def test_a_path_does_not_match_a_different_route():
    assert not paths_match("/audio/start", "/audio/stop")
    assert not paths_match("/settings", "/settings/storage")
