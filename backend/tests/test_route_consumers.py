"""A backend route is live only if something names its path.

``[tool.vulture]`` in ``backend/pyproject.toml`` carries
``ignore_decorators = ["@router.*"]``, which is what makes ``min_confidence = 60``
affordable (ADR 051), and the exemption is structural: a FastAPI handler nobody
calls is exempt exactly like a live one. ``src/api-surface.test.ts`` closes only
the other half of the pair -- it reports an ``api`` member with no caller, never
a backend route with no client. This module closes the remaining half. ADR 054
records the design and its rejected alternatives.

**What it sees.** Every route registered on ``app.main.app`` whose endpoint is
not defined inside FastAPI itself, matched by path against the string and
template literals scanned out of ``src/api.ts``. Paths arrive from FastAPI with
every router prefix already applied, which is why the gate lives on the Python
side: prefixes are declared both at ``app.include_router(...)`` and on
``APIRouter(prefix=...)``, so reading decorator strings would yield the wrong
path for six of the seven routers.

**What it cannot see, stated rather than implied.**

* **Verb-level deadness.** A client literal matches a route by path alone. Two
  application paths carry two verbs each (``/history`` GET and DELETE,
  ``/settings`` GET and PUT); for those, one verb losing its caller is
  invisible. ``src/api.ts`` puts the verb in three different positions relative
  to the path, so every pairing rule covering all three is positional and
  approximate.
* **A path the client never spells as a literal.** One assembled by
  concatenation makes its route look dead -- a loud failure toward an exemption
  rather than a silent pass.
* **A path literal held by an unexported, uncalled helper inside
  ``src/api.ts``.** knip does not report unexported symbols and
  ``tsconfig.json`` sets ``strict`` without ``noUnusedLocals``, so nothing in
  the repository would report such a helper.

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

import re
from pathlib import Path

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]

API_TS = REPO_ROOT / "src" / "api.ts"
API_SURFACE_TEST_TS = REPO_ROOT / "src" / "api-surface.test.ts"
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"
SRC_DIR = REPO_ROOT / "src"
APP_DIR = REPO_ROOT / "backend" / "app"

INTERPOLATION_WILDCARD = "\x00interpolation\x00"

CONSUMED_OUTSIDE_TYPESCRIPT: dict[str, tuple[tuple[Path, str], ...]] = {
    "/shutdown": ((BACKEND_RS, r'format!\("http://127\.0\.0\.1:\{\}/shutdown", PORT\)'),),
}
"""Routes kept alive by a caller outside ``src/``, each site named with the
regular expression that file must still match.

A regular expression rather than a substring search: ``/shutdown`` also appears
in doc comments in the same Rust file (``src-tauri/src/backend.rs:972``,
``:990``) that call nothing, so a substring search would pass on a file whose
only real call site had been deleted. This is the declared-sites shape
``test_cross_language_contracts.py`` uses.

``/health`` needs no entry despite its Rust (``src-tauri/src/backend.rs:838``)
and Python (``backend/scripts/smoke_sidecar.py:94``) consumers, because
``src/api.ts:644`` already calls it -- and an entry here whose route turns out
to have a client literal after all is reported as redundant below.
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
callers dead. Their routes are subtracted from the consumed set before the main
assertion runs, and the key set is asserted equal to the array read out of
``src/api-surface.test.ts`` so the two gates cannot drift apart.
"""

ROUTER_DECORATOR = re.compile(r"@router\.(?:get|post|put|delete|patch)\(")

TYPESCRIPT_ALLOWLIST = re.compile(
    r"const\s+ALLOWED_WITHOUT_PRODUCTION_CALLER\s*=\s*\[(.*?)\]",
    re.DOTALL,
)


def application_route_registrations() -> list[tuple[str, frozenset[str]]]:
    """Every route registration whose endpoint is defined under the ``app``
    package, as a path and the HTTP verbs that registration carries.

    Origin, not a name list: FastAPI's own ``/docs``, ``/docs/oauth2-redirect``,
    ``/redoc`` and ``/openapi.json`` are defined inside ``fastapi`` and are
    dropped by the rule, so a fifth built-in needs no edit here.

    The rule names what is dropped rather than what is kept, and that direction
    is load-bearing. Keeping endpoints whose ``__module__`` starts with ``app.``
    assumes the application is imported under that exact top-level name; under
    an editable install on CI it need not be, and the whole set then filters
    away, leaving every assertion below trivially true over nothing. Dropping
    FastAPI's own is the half that is certain wherever the package is imported
    from.

    One entry per registration rather than per path, because the decorator
    cross-check below counts declaration sites: two registrations of the same
    path collapse into one key and would leave that check reading a count it
    could not distinguish from a missing route.
    """
    registrations: list[tuple[str, frozenset[str]]] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)
        if methods is None or endpoint is None:
            continue
        if getattr(endpoint, "__module__", "").startswith("fastapi."):
            continue
        registrations.append((route.path, frozenset(methods - {"HEAD", "OPTIONS"})))
    return registrations


def application_routes() -> dict[str, frozenset[str]]:
    """Application route paths, each mapped to every verb registered on it."""
    routes: dict[str, set[str]] = {}
    for path, verbs in application_route_registrations():
        routes.setdefault(path, set()).update(verbs)
    return {path: frozenset(verbs) for path, verbs in routes.items()}


def scan_string_literals(source: str) -> tuple[list[str], str]:
    """Every string, single-quoted and template literal in ``source``, plus the
    mode the scanner ended in.

    Comments are skipped *before* quotes are honoured. ``src/api.ts`` carries
    prose apostrophes inside ``/** ... */`` blocks (``Tauri's``, ``ADR 049's``),
    and a scanner that enters single-quote mode on one of them desynchronises
    and silently loses every literal after it.

    A template literal is read whole, interpolations included: its contents are
    matched as a pattern later rather than parsed here, so a ``{``, ``)`` or
    ``;`` inside one is data. An escaped character inside any literal is
    consumed as a pair, so ``\\"`` does not end the literal.

    The returned mode is the precise statement of "the scanner read the whole
    file rather than a prefix": anything but ``code`` means an unterminated
    literal or comment swallowed the tail.
    """
    literals: list[str] = []
    current: list[str] = []
    mode = "code"
    index = 0

    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if mode == "code":
            if char == "/" and following == "/":
                mode = "line-comment"
                index += 2
            elif char == "/" and following == "*":
                mode = "block-comment"
                index += 2
            elif char in ("'", '"', "`"):
                mode = char
                current = []
                index += 1
            else:
                index += 1
            continue

        if mode == "line-comment":
            if char == "\n":
                mode = "code"
            index += 1
            continue

        if mode == "block-comment":
            if char == "*" and following == "/":
                mode = "code"
                index += 2
            else:
                index += 1
            continue

        if char == "\\":
            current.append(source[index : index + 2])
            index += 2
            continue
        if char == mode:
            literals.append("".join(current))
            mode = "code"
            index += 1
            continue
        current.append(char)
        index += 1

    return literals, mode


def client_path_literals(source: str) -> set[str]:
    """The literals from ``source`` that look like a backend path.

    A leading ``/`` and not ``//``: relative imports start with ``.`` and URLs
    with ``h``, so the filter yields no false candidates on ``src/api.ts``.
    """
    literals, _ = scan_string_literals(source)
    return {value for value in literals if value.startswith("/") and not value.startswith("//")}


def normalise_path(path: str) -> tuple[str, ...]:
    """``path`` as comparable segments.

    Each ``${...}`` interpolation collapses to a wildcard token first, so a
    ``?`` appearing inside an interpolated expression cannot be mistaken for the
    start of the query string; then everything from the first ``?`` is dropped.
    """
    collapsed = re.sub(r"\$\{.*?\}", INTERPOLATION_WILDCARD, path)
    return tuple(collapsed.split("?", 1)[0].split("/"))


def paths_match(route_path: str, literal: str) -> bool:
    """Whether ``literal`` addresses ``route_path``.

    Segment by segment over lists of equal length. A FastAPI ``{param}``
    placeholder matches any literal segment, and an interpolation inside a
    literal segment matches any run of characters *within* that segment -- the
    two rules are independent, since either side alone can be the variable one.
    This resolves ``/history/${id}`` against ``/history/{entry_id}`` and
    ``/words/top?lang=${lang}`` against ``/words/top``.

    The interpolation is a bounded wildcard rather than a whole-segment one on
    purpose. Letting a segment that merely contains an interpolation match
    anything makes ``/history?limit=${limit}&…`` certify ``/shutdown`` as
    consumed, which is a silent false negative in the one direction this gate
    exists to catch.
    """
    route_segments = normalise_path(route_path)
    literal_segments = normalise_path(literal)
    if len(route_segments) != len(literal_segments):
        return False
    return all(
        (route_segment.startswith("{") and route_segment.endswith("}"))
        or _segment_matches(route_segment, literal_segment)
        for route_segment, literal_segment in zip(route_segments, literal_segments)
    )


def _segment_matches(route_segment: str, literal_segment: str) -> bool:
    """Whether one path segment of a client literal addresses one route segment."""
    if INTERPOLATION_WILDCARD not in literal_segment:
        return route_segment == literal_segment
    pattern = ".*".join(re.escape(part) for part in literal_segment.split(INTERPOLATION_WILDCARD))
    return re.fullmatch(pattern, route_segment) is not None


def consumed_routes() -> set[str]:
    """Route paths a live client literal addresses.

    The routes named by ``CLIENT_MEMBERS_WITHOUT_CALLER`` are subtracted: their
    literals sit inside ``api`` members the neighbouring gate already reports as
    having no production caller, so a literal there does not make a route live.

    That subtraction is by path rather than per literal, which is a stated limit:
    a second, genuinely called member whose literal collapsed to the same path as
    a caller-less one would be subtracted with it, and the route would then demand
    an allowlist entry it does not need. No path is reachable that way today --
    each entry's path is addressed by exactly one literal -- and the failure is a
    false alarm asking for a rule, never a route waved through.
    """
    literals = client_path_literals(API_TS.read_text(encoding="utf-8"))
    matched = {
        path
        for path in application_routes()
        if any(paths_match(path, literal) for literal in literals)
    }
    return matched - set(CLIENT_MEMBERS_WITHOUT_CALLER.values())


def typescript_allowlist_members() -> list[str]:
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
    return re.findall(r'"([^"]*)"', match.group(1))


def production_typescript_modules() -> dict[Path, str]:
    """Every non-test TypeScript module under ``src/``, by path."""
    return {
        path: path.read_text(encoding="utf-8")
        for path in sorted(SRC_DIR.rglob("*.ts"))
        if not path.name.endswith(".test.ts")
    }


def test_the_scanner_reads_api_ts_to_its_end():
    literals, final_mode = scan_string_literals(API_TS.read_text(encoding="utf-8"))

    assert final_mode == "code", (
        f"the literal scanner ended {API_TS.name} in `{final_mode}` mode, so it read a "
        f"prefix of the file and stopped. Every path literal after the cut is missing and "
        f"every assertion in this module is silently incomplete"
    )
    assert literals, f"the scanner found no string literals at all in {API_TS.name}"


def test_every_application_route_has_a_consumer():
    routes = application_routes()

    assert routes, (
        "app.main.app registered no application route at all, so every assertion in "
        "this module is trivially true over an empty set. The enumeration rule in "
        "application_route_registrations() no longer matches how this application is "
        "imported here -- it is the gate that is broken, not the routes"
    )

    unconsumed = (
        set(routes)
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
    routes = application_routes()
    literals = client_path_literals(API_TS.read_text(encoding="utf-8"))
    orphans = sorted(
        literal
        for literal in literals
        if not any(paths_match(path, literal) for path in routes)
    )

    assert orphans == [], (
        f"these path literals in src/api.ts address no backend route: {orphans}. The "
        f"client is pointing at a renamed or deleted endpoint and would fail at runtime"
    )


def test_the_route_set_matches_the_router_decorator_sites():
    decorator_sites = sum(
        len(ROUTER_DECORATOR.findall(path.read_text(encoding="utf-8")))
        for path in sorted(APP_DIR.rglob("*.py"))
    )

    registered_verbs = sum(len(verbs) for _, verbs in application_route_registrations())

    assert decorator_sites == registered_verbs, (
        f"{decorator_sites} `@router.<verb>` decorator sites under backend/app do not "
        f"account for the {registered_verbs} verbs app.main.app registers. A route "
        f"reached this application by some other mechanism -- registered directly on the "
        f"app object, "
        f"or added with app.add_api_route -- and this module's rule has to be extended to "
        f"see it before it can claim to check every route"
    )


def test_no_declared_outside_consumer_has_outlived_its_route():
    routes = application_routes()
    stale = sorted(path for path in CONSUMED_OUTSIDE_TYPESCRIPT if path not in routes)

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
    routes = application_routes()
    stale = sorted(path for path in UNCONSUMED_PENDING_A_DECISION if path not in routes)

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
    routes = application_routes()
    stale = sorted(
        f"{member} -> {path}"
        for member, path in CLIENT_MEMBERS_WITHOUT_CALLER.items()
        if path not in routes
    )

    assert stale == [], (
        f"these CLIENT_MEMBERS_WITHOUT_CALLER entries name routes that no longer exist: "
        f"{stale}. The mapping subtracts these paths from the consumed set, so a stale one "
        f"subtracts nothing and hides its member's real route"
    )


BACKEND_ADDRESS_TOKENS = ("BACKEND_BASE_URL", "127.0.0.1", "localhost")


def _addresses_the_backend(source: str) -> bool:
    """Whether a module reaches the backend at all, however it names the host.

    ``BACKEND_BASE_URL`` alone would miss a module that hardcodes the host
    instead of importing the constant -- the seam ``docs/style-guide.md`` already
    names -- and that module's routes would be reported dead by this file.
    """
    return "fetch(" in source and any(
        token in source for token in BACKEND_ADDRESS_TOKENS
    )


def test_api_ts_is_the_whole_client_surface():
    others = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path, source in production_typescript_modules().items()
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


def test_a_prose_apostrophe_in_a_line_comment_does_not_desynchronise_the_scanner():
    literals, final_mode = scan_string_literals(SYNTHETIC_LINE_COMMENT)

    assert final_mode == "code", (
        f"an apostrophe inside a `//` comment left the scanner in `{final_mode}` mode. "
        f"On src/api.ts that means silently losing every path literal after the first "
        f"piece of prose that contains one"
    )
    assert literals == ["/plain"]


def test_a_prose_apostrophe_in_a_block_comment_does_not_desynchronise_the_scanner():
    literals, final_mode = scan_string_literals(SYNTHETIC_BLOCK_COMMENT)

    assert final_mode == "code", (
        f"an apostrophe inside a `/** */` block left the scanner in `{final_mode}` mode. "
        f"src/api.ts carries `Tauri's` and `ADR 049's` in exactly such blocks"
    )
    assert literals == ["/plain"]


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


def test_an_interpolated_segment_matches_a_route_parameter():
    assert paths_match("/history/{entry_id}", "/history/${entry.id}")
    assert not paths_match("/history/{entry_id}", "/history/${entry.id}/extra")


def test_a_route_parameter_matches_a_plain_literal_segment():
    assert paths_match("/history/{entry_id}", "/history/42")


def test_an_interpolation_matches_a_fixed_route_segment():
    assert paths_match("/audio/level-stream", "/audio/${kind}-stream")
    assert not paths_match("/audio/level-stream-extra", "/audio/${kind}-stream"), (
        "an interpolated segment matched a route segment it is only a prefix of. The "
        "wildcard has to span the whole segment, or every route sharing a prefix with a "
        "consumed one is certified live by association"
    )


def test_a_query_string_is_dropped_at_the_first_question_mark():
    assert paths_match("/words/top", "/words/top?lang=${lang}&limit=${limit}")
    assert paths_match("/history", "/history?limit=${limit}&before_ts=${cursor.ts}")


def test_a_path_does_not_match_a_different_route():
    assert not paths_match("/audio/start", "/audio/stop")
    assert not paths_match("/settings", "/settings/storage")
