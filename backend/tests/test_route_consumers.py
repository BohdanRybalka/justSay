"""A backend route is live only if something names its path.

``[tool.vulture]`` in ``backend/pyproject.toml`` carries
``ignore_decorators = ["@router.*"]``, which is what makes ``min_confidence = 60``
affordable (ADR 051), and the exemption is structural: a FastAPI handler nobody
calls is exempt exactly like a live one. ``src/api-surface.test.ts`` closes only
the other half of the pair -- it reports an ``api`` member with no caller, never
a backend route with no client. This module closes the remaining half. ADR 054
records the design.

**What it does.** Takes every path FastAPI publishes in its own OpenAPI schema
and asks whether that path is written anywhere in ``src/api.ts``, as text. That
is the whole rule.

**Why a text search and not a parser.** The first version of this module lexed
TypeScript from Python to attribute path literals to ``api`` members and match
them against route templates. Three review rounds found four soundness holes in
it, each one letting a dead route be certified live -- the failure direction that
makes a gate worse than none. The rewrite is the user's decision of 2026-09-09,
taken on a measurement rather than a preference: the text rule reports exactly
the same three unconsumed routes on the current tree, at a fortieth of the size,
and its own failure direction is the safe one. Of 27 routes exactly one carries a
path parameter, so the machinery served a single case.

**What it cannot see, stated rather than implied.**

* **A path named only in a comment or a dead helper** counts as named. This gate
  can therefore miss a dead route; it cannot report a live one dead. The other
  half of that question -- whether the ``api`` member holding the literal has a
  caller at all -- is ``src/api-surface.test.ts``'s, and neither gate is
  sufficient alone.
* **Verb-level deadness.** ``/history`` carries GET and DELETE, ``/settings`` GET
  and PUT; one verb losing its caller is invisible here.
* **A path the client assembles by concatenation** looks unnamed, which is a loud
  failure toward an allowlist entry rather than a silent pass.
* **A route excluded from the schema** would be invisible, which is why
  ``test_no_route_hides_itself_from_the_schema`` forbids ``include_in_schema``
  outright rather than trusting that nobody adds it.

**Why the schema and not ``app.routes``.** ``app.routes`` is internal structure
and it moved: a FastAPI newer than the 0.135.3 on the machine this was written on
stops flattening ``include_router`` and nests children inside an
``_IncludedRouter``, so the walk found 33 entries locally and 11 on CI, and every
assertion here passed over the empty set that produced. The schema is the public
surface the application actually serves, with every router prefix applied. The
unpinned dependency behind that difference is JS-141.

**The two allowlists.** ``CONSUMED_OUTSIDE_TYPESCRIPT`` names a route kept alive
from outside ``src/`` together with the regular expression its consumer file must
still match -- a regular expression rather than a substring because ``/shutdown``
also appears in Rust doc comments that call nothing, so a substring search would
pass on a file whose only real call site had been deleted. ``/health`` needs no
entry despite its Rust and Python consumers, because ``src/api.ts`` names it, and
a redundant entry is reported. ``UNCONSUMED_PENDING_A_DECISION`` names a route no
client names, with the decision that would remove the entry; it is a record
rather than a suppression, failing both when its route disappears and when its
path acquires a client literal. ``/stt/local/load`` and ``/stt/local/unload`` are
deliberately absent -- ``src/api.ts`` does name them, so this gate has nothing to
say about them, and whether their members have callers is
``src/api-surface.test.ts``'s question, which it already answers.

**Why every file it reads is named explicitly.** ``backend/build/`` and
``backend/.venv-build/Lib/site-packages/app/`` hold stale full copies of ``app/``
on any machine that has run an editable install, so a glob from the repository
root would silently measure the wrong tree. The one glob used, ``backend/app``,
is inside the package itself.
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
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"
APP_DIR = REPO_ROOT / "backend" / "app"

HTTP_VERB_DECORATORS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})

LITERAL_OPENER = r"([\"'`])"
LITERAL_TERMINATOR = r"(?:\1|[?&#])"
INTERPOLATED_SEGMENT = r"\$\{[^}]*\}"

CONSUMED_OUTSIDE_TYPESCRIPT: dict[str, tuple[tuple[Path, str], ...]] = {
    "/shutdown": ((BACKEND_RS, r'format!\("http://127\.0\.0\.1:\{\}/shutdown", PORT\)'),),
}

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
}


class RouterDecorator(NamedTuple):
    """One ``@router.<attribute>(...)`` decorator found under ``backend/app``."""

    module: str
    attribute: str
    keywords: tuple[str, ...]


@functools.cache
def api_ts_source() -> str:
    return API_TS.read_text(encoding="utf-8")


@functools.cache
def application_schema_paths() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Every path in the served OpenAPI schema, with the verbs declared on it.

    Verbs are filtered against ``HTTP_VERB_DECORATORS`` rather than taken as the
    whole key set, because a path item also carries non-operation keys.
    """
    return tuple(
        (path, tuple(sorted(key.upper() for key in item if key in HTTP_VERB_DECORATORS)))
        for path, item in sorted(app.openapi()["paths"].items())
    )


@functools.cache
def application_paths() -> frozenset[str]:
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


def route_is_named_in(route_path: str, source: str) -> bool:
    """Whether ``source`` writes ``route_path`` as the path of a request.

    The path must sit inside a string or template literal and occupy the whole of
    it up to a query separator: a quote or backtick immediately before it, and
    either that same quote or a ``?``, ``&`` or ``#`` immediately after.

    **All three conditions are load-bearing and each was added after a mutation
    proved the others alone are not enough.** Without the closing one, a deleted
    ``/history`` reads as live because ``/history/stats`` contains it. Without
    the opening one, a deleted ``/cloud-status`` reads as live because
    ``/settings/cloud-status`` ends with it -- a route being a text suffix of an
    unrelated live path is not rare, it is what a shared final segment looks
    like. Without requiring the closing quote to be the one that opened, an
    escaped quote inside a string can be read as an opener.

    A path parameter is matched as an interpolation in the same position rather
    than by truncating the path at the placeholder: ``/history/{entry_id}`` is
    named by ``/history/${...}`` and by nothing else. Truncating instead would
    let ``/{entry_id}/dead-canary`` be certified live by any literal beginning
    with an interpolated first segment.
    """
    parts = re.split(r"\{[^}]*\}", route_path)
    body = INTERPOLATED_SEGMENT.join(re.escape(part) for part in parts)
    return re.search(LITERAL_OPENER + body + LITERAL_TERMINATOR, source) is not None


@functools.cache
def unnamed_routes() -> frozenset[str]:
    source = api_ts_source()
    return frozenset(path for path in application_paths() if not route_is_named_in(path, source))


def test_every_application_route_is_named_by_the_client():
    routes = application_paths()

    assert routes, (
        "app.main.app publishes no path at all, so every assertion in this module is "
        "trivially true over an empty set. The enumeration in application_schema_paths() "
        "no longer matches how this application is built here -- it is the gate that is "
        f"broken, not the routes. The schema holds {len(app.openapi()['paths'])} paths"
    )

    unconsumed = sorted(
        unnamed_routes() - set(CONSUMED_OUTSIDE_TYPESCRIPT) - set(UNCONSUMED_PENDING_A_DECISION)
    )

    assert unconsumed == [], (
        f"these backend routes are named nowhere in src/api.ts: {unconsumed}. Either "
        f"delete the route and its client half together (the handler, its response model "
        f"if nothing else uses it, and its test), or add an entry with its reason: "
        f"CONSUMED_OUTSIDE_TYPESCRIPT if a caller outside src/ keeps it alive, "
        f"UNCONSUMED_PENDING_A_DECISION if it is waiting on a decision (ADR 054)"
    )


def test_the_route_set_matches_the_router_decorator_sites():
    decorator_sites = sum(
        1 for decorator in router_decorators() if decorator.attribute in HTTP_VERB_DECORATORS
    )
    registered_verbs = sum(len(verbs) for _, verbs in application_schema_paths())

    assert decorator_sites == registered_verbs, (
        f"{decorator_sites} `@router.<verb>` decorator sites under backend/app do not "
        f"account for the {registered_verbs} verbs app.main.app publishes. Either a route "
        f"reached this application by another mechanism -- registered on the app object, "
        f"or added with app.add_api_route -- or a router module is never included, or a "
        f"route was hidden from the schema. This module's rule has to be extended to see "
        f"it before it can claim to check every route"
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
    stale = sorted(set(CONSUMED_OUTSIDE_TYPESCRIPT) - application_paths())

    assert stale == [], (
        f"these CONSUMED_OUTSIDE_TYPESCRIPT entries name routes that no longer exist: "
        f"{stale}. A stale entry silently exempts whatever future route reuses the path"
    )


def test_every_declared_outside_consumer_site_still_calls_its_route():
    broken = sorted(
        f"{route} -> {path.relative_to(REPO_ROOT).as_posix()}"
        for route, sites in CONSUMED_OUTSIDE_TYPESCRIPT.items()
        for path, pattern in sites
        if not re.search(pattern, path.read_text(encoding="utf-8"))
    )

    assert broken == [], (
        f"these declared consumer sites no longer match: {broken}. The route is exempt "
        f"from the consumer check on the strength of a call that is not there any more"
    )


def test_no_declared_outside_consumer_is_redundant():
    redundant = sorted(set(CONSUMED_OUTSIDE_TYPESCRIPT) - unnamed_routes())

    assert redundant == [], (
        f"these CONSUMED_OUTSIDE_TYPESCRIPT entries name routes src/api.ts already names: "
        f"{redundant}. The exemption is buying nothing and hides the client half from view"
    )


def test_no_pending_decision_entry_has_outlived_its_route():
    stale = sorted(set(UNCONSUMED_PENDING_A_DECISION) - application_paths())

    assert stale == [], (
        f"these UNCONSUMED_PENDING_A_DECISION entries name routes that no longer exist: "
        f"{stale}. The decision was taken and the route deleted; delete its entry too, or "
        f"a future route reusing the path inherits the exemption"
    )


def test_no_pending_decision_entry_has_acquired_a_consumer():
    consumed = sorted(set(UNCONSUMED_PENDING_A_DECISION) - unnamed_routes())

    assert consumed == [], (
        f"these UNCONSUMED_PENDING_A_DECISION entries name routes src/api.ts now names: "
        f"{consumed}. The decision they were waiting on has been taken -- delete the entry "
        f"so the route is checked like every other one"
    )


SYNTHETIC_CLIENT = "\n".join(
    [
        'const a = request("GET", "/plain");',
        "const b = request(`GET`, `/query?limit=${limit}`);",
        "const c = request(`DELETE`, `/thing/${id}`);",
        "const d = request(`GET`, `/prefix/longer`);",
        'const e = request("GET", "/scoped/tail");',
        "// /commented is mentioned here and called nowhere.",
    ]
)


def test_a_static_path_written_whole_is_named():
    assert route_is_named_in("/plain", SYNTHETIC_CLIENT)


def test_a_static_path_followed_by_a_query_string_is_named():
    assert route_is_named_in("/query", SYNTHETIC_CLIENT)


def test_a_parameterised_route_is_named_by_its_prefix_and_an_interpolation():
    assert route_is_named_in("/thing/{thing_id}", SYNTHETIC_CLIENT)


def test_a_prefix_of_a_longer_literal_is_not_named_by_it():
    assert not route_is_named_in("/prefix", SYNTHETIC_CLIENT), (
        "`/prefix` was read as named by the `/prefix/longer` literal, so deleting the "
        "client half of a route whose path is a prefix of a sibling's would go unreported"
    )


def test_a_sibling_under_a_parameterised_prefix_is_not_named_by_the_interpolation():
    assert not route_is_named_in("/thing/canary-dead", SYNTHETIC_CLIENT), (
        "`/thing/canary-dead` was read as named by the `/thing/${id}` literal. That is the "
        "hole three review rounds found in the parser this rule replaced: one interpolated "
        "literal certified every sibling under the same prefix as live"
    )


def test_a_suffix_of_a_longer_path_is_not_named_by_it():
    assert not route_is_named_in("/tail", SYNTHETIC_CLIENT), (
        "`/tail` was read as named by the `/scoped/tail` literal. A route being a text "
        "suffix of an unrelated live path is what a shared final segment looks like, so "
        "without the opening boundary a deleted route of that shape reads as live"
    )


def test_a_parameterised_route_is_not_named_by_a_different_shape():
    assert not route_is_named_in("/{thing_id}/dead-canary", SYNTHETIC_CLIENT), (
        "`/{thing_id}/dead-canary` was read as named by the `/thing/${id}` literal. That "
        "happens when a parameterised path is truncated at its placeholder instead of "
        "matched in position, and it certifies every route with an interpolated first "
        "segment as live"
    )


def test_a_path_closed_by_a_different_quote_is_not_named():
    assert not route_is_named_in("/x", 'const a = "/x`;'), (
        "`/x` was read as named by text that opens with one quote and closes with "
        "another, so an escaped quote inside a string can act as an opener and certify "
        "a dead route"
    )


def test_a_path_nobody_writes_is_not_named():
    assert not route_is_named_in("/absent", SYNTHETIC_CLIENT)


def test_a_path_at_the_very_end_of_a_source_does_not_raise():
    assert not route_is_named_in("/tail", "const truncated = \"/tail")
