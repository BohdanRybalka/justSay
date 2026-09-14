"""A client path is live only if the backend still serves it.

``backend/tests/test_route_consumers.py`` closes one direction of the pair:
every path FastAPI publishes is written somewhere in ``src/api.ts``. This module
closes the other one. Nothing reported a request the application still makes to
a path the backend no longer serves -- the first report was a user's 404. ADR
062 records the design.

**What it does.** Parses ``src/api.ts`` with the TypeScript compiler API,
reconstructs the text of every string and template literal, keeps the ones that
look like a request path, and asks whether each addresses a path in the served
OpenAPI schema.

**Why a parse and not a text search.** This file's own comments name paths --
prose about ``/health``, a JSDoc block naming an endpoint -- so a pattern over
the source text reports a path the program never requests. The first attempt at
this check lexed TypeScript by hand and three review rounds found four soundness
holes in it; a regular expression is worse, not better. A comment is trivia and
never a node, so a parser skips one without being told to. The tree comes from
``backend/tests/extract_client_paths.mjs``, run as a subprocess, because the
TypeScript compiler is already installed in this repository and reimplementing
its lexer in Python is the machinery that failed.

**What it cannot see, stated rather than implied.**

* **A path this application never writes whole.** ``.join("/")``, ``new URL()``,
  a helper that returns a path, a path constant imported from another module:
  the reconstruction rule sees none of them.
  ``test_no_client_path_is_assembled`` catches the two shapes that do occur --
  a ``+`` concatenation and a template with a leading interpolation -- and
  ``test_api_ts_is_the_whole_client_surface`` confines any other construction to
  the one file a reader can check. Nothing reports the rest, and that is the
  residual hole.
* **Verb-level mismatch.** ``/history`` carries GET and DELETE and ``/settings``
  GET and PUT; a client asking the wrong verb on a live path passes here.
* **A route hidden from the schema** would make its client literal read as dead.
  ``test_route_consumers.py``'s ``test_no_route_hides_itself_from_the_schema``
  forbids ``include_in_schema`` outright, which is what keeps that from
  happening -- but the guarantee lives in that file, not this one.
* **A path-shaped literal that is not a request.** The one heuristic here is the
  whitespace clause: a ``/``-leading literal with no whitespace in it is taken
  for a request path. Zero of the 28 literals on the current tree are anything
  else. The first one that is reddens this gate, and the remedies are changing
  the literal or adding an exemption table under review.

**Why the schema and not ``app.routes``.** The same reason its sibling gives:
``app.routes`` is internal structure that moved between FastAPI versions and
produced a different count locally and on CI, with every assertion passing over
the empty set that left. The schema is the surface the application serves.

**Why it fails rather than skips when its tooling is missing.** A gate that
skips when ``node`` or ``node_modules/typescript`` is absent is green on the day
its tooling breaks, which is the failure direction that makes a gate worse than
none. Every such condition raises with ``npm install`` named first.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]

API_TS = REPO_ROOT / "src" / "api.ts"
SRC_DIR = REPO_ROOT / "src"
EXTRACTOR = Path(__file__).resolve().parent / "extract_client_paths.mjs"

INTERPOLATION = "\x00"
PARAMETER = "*"

TOOLING_REMEDY = (
    "run `npm install` in the repository root: this gate parses src/api.ts with the "
    "TypeScript compiler already listed in package.json, and it fails rather than skips "
    "when that tooling is missing, because a gate that skips is green on the day it breaks"
)

ROUTES_NO_CLIENT_LITERAL_NAMES = frozenset({"/audio/stop", "/shutdown", "/stt/local/install"})


class ClientLiteral(NamedTuple):
    """One reconstructed literal that looks like a request path."""

    line: int
    text: str


class AssembledPath(NamedTuple):
    """One path built out of parts rather than written whole."""

    line: int
    text: str
    shape: str


class FetchCall(NamedTuple):
    """The shape of the first argument of one ``fetch()`` call."""

    line: int
    head_is_empty: bool
    span_expression_kinds: tuple[str, ...]
    span_identifiers: tuple[str | None, ...]
    span_literal_texts: tuple[str, ...]


class ExtractorOutput(NamedTuple):
    """Everything ``extract_client_paths.mjs`` reports about one source file."""

    literals: tuple[ClientLiteral, ...]
    assembled: tuple[AssembledPath, ...]
    fetch_calls: tuple[FetchCall, ...]


def run_extractor(source: Path) -> ExtractorOutput:
    """Parse ``source`` with the TypeScript compiler and read back its literals.

    Raises rather than skips on every failure: no ``node`` on PATH, a non-zero
    exit (which is what a file that does not parse produces, carrying the
    compiler's own diagnostics), or output that is not JSON.
    """
    node = shutil.which("node")
    assert node is not None, (
        f"node is not on PATH, so src/api.ts cannot be parsed and this gate can say "
        f"nothing about the client's paths -- {TOOLING_REMEDY}"
    )

    completed = subprocess.run(
        [node, str(EXTRACTOR), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{EXTRACTOR.name} exited {completed.returncode} on {source}. If it names a parse "
        f"diagnostic, the TypeScript file is broken and no literal could be read from it -- "
        f"a client that does not parse is not a client with no dead paths. If it names a "
        f"missing module, {TOOLING_REMEDY}.\nstderr:\n{completed.stderr}"
    )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{EXTRACTOR.name} printed something other than JSON on {source} ({error}) -- "
            f"{TOOLING_REMEDY}.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        ) from error

    return ExtractorOutput(
        literals=tuple(ClientLiteral(item["line"], item["text"]) for item in payload["literals"]),
        assembled=tuple(
            AssembledPath(item["line"], item["text"], item["shape"])
            for item in payload["assembled"]
        ),
        fetch_calls=tuple(
            FetchCall(
                line=item["line"],
                head_is_empty=item["headIsEmpty"],
                span_expression_kinds=tuple(item["spanExpressionKinds"]),
                span_identifiers=tuple(item["spanIdentifiers"]),
                span_literal_texts=tuple(item["spanLiteralTexts"]),
            )
            for item in payload["fetchCalls"]
        ),
    )


@functools.cache
def client_paths() -> ExtractorOutput:
    return run_extractor(API_TS)


@functools.cache
def synthetic_paths() -> ExtractorOutput:
    """The same extractor over ``SYNTHETIC_CLIENT``, so each rule has its own pin.

    The synthetic source is written to a temporary file rather than passed on
    stdin so that it travels the exact path ``src/api.ts`` travels.
    """
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "synthetic.ts"
        source.write_text(SYNTHETIC_CLIENT, encoding="utf-8")
        return run_extractor(source)


def route_segments(path: str) -> tuple[str, ...]:
    """A schema path as segments, with every ``{name}`` placeholder as ``*``."""
    return tuple(
        PARAMETER if segment.startswith("{") and segment.endswith("}") else segment
        for segment in path.split("/")
    )


def client_segments(text: str) -> tuple[str, ...]:
    """A client literal as segments, with every whole interpolation as ``*``.

    The text is truncated at the first ``?`` or ``#`` because everything after
    one addresses the same path. A segment counts as a parameter only when it is
    *exactly* one interpolation: ``/${x}ealth`` keeps its segment verbatim and
    matches nothing, and ``/history/42`` does not match ``/history/{entry_id}``.
    Truncating a parameterised path at its placeholder instead -- the defect
    ``test_route_consumers.py`` pins in the mirror direction -- would let one
    interpolated literal certify every sibling under the same prefix.
    """
    for separator in ("?", "#"):
        text = text.split(separator, 1)[0]
    return tuple(PARAMETER if segment == INTERPOLATION else segment for segment in text.split("/"))


@functools.cache
def application_paths() -> frozenset[str]:
    return frozenset(app.openapi()["paths"])


@functools.cache
def addressed_route_shapes() -> frozenset[tuple[str, ...]]:
    return frozenset(client_segments(literal.text) for literal in client_paths().literals)


@functools.cache
def unaddressed_literals() -> tuple[ClientLiteral, ...]:
    live = {route_segments(path) for path in application_paths()}
    return tuple(
        literal for literal in client_paths().literals if client_segments(literal.text) not in live
    )


def test_the_extractor_read_a_non_empty_client_surface():
    output = client_paths()

    assert output.literals, (
        f"{EXTRACTOR.name} read no path literal at all from {API_TS}, so every assertion "
        f"in this module is trivially true over an empty set. It is the gate that is "
        f"broken, not the client: either the reconstruction rule no longer matches how "
        f"this file writes its paths, or the parse produced a tree of error nodes. "
        f"{API_TS.name} is {len(API_TS.read_text(encoding='utf-8').splitlines())} lines long"
    )


def test_every_client_path_literal_addresses_a_live_route():
    unaddressed = sorted(
        f"src/api.ts:{literal.line}: {literal.text!r}" for literal in unaddressed_literals()
    )

    assert unaddressed == [], (
        f"these paths are requested by src/api.ts and served by no route in the "
        f"application's OpenAPI schema: {unaddressed}. Either the route was renamed and "
        f"the client half was left behind -- fix the literal -- or the route was deleted "
        f"and its client half should have gone with it: delete the api member, its "
        f"callers and its response type. A path parameter is matched in position, so "
        f"`/history/42` is a different path from `/history/{{entry_id}}` on purpose"
    )


def test_no_client_path_is_assembled():
    findings = sorted(
        f"src/api.ts:{item.line}: {item.text!r} ({item.shape})" for item in client_paths().assembled
    )

    assert findings == [], (
        f"these request paths are built out of parts rather than written whole: "
        f"{findings}. The reconstruction rule cannot read the path such a shape produces, "
        f"so the check above would pass over it silently. Write the path as one literal"
    )


def test_every_backend_request_is_composed_from_the_base_url_and_a_path():
    """Both ``fetch`` calls put the base URL first and write no path inline.

    This pins the *composition*, not the provenance: it says the path arrives as
    an interpolated expression after ``BACKEND_BASE_URL`` and never as text
    inside the template, which is what makes the literal rule above sufficient.
    It says nothing about where that expression got its value.
    """
    calls = client_paths().fetch_calls
    shapes = sorted(
        (
            call.head_is_empty,
            call.span_expression_kinds,
            call.span_identifiers[:1],
            call.span_literal_texts,
        )
        for call in calls
    )
    composed = (True, ("Identifier", "Identifier"), ("BACKEND_BASE_URL",), ("", ""))

    assert shapes == [composed, composed], (
        f"the fetch calls in src/api.ts no longer have the shape this gate relies on: "
        f"{shapes}, at lines {sorted(call.line for call in calls)}. Every one must be a "
        f"template with an empty head whose first interpolation is BACKEND_BASE_URL and "
        f"whose literal parts are all empty. A path written inside the template instead "
        f"is a path the literal rule cannot see"
    )


def test_the_unnamed_routes_are_the_three_the_sibling_gate_allowlists():
    addressed = addressed_route_shapes()
    unnamed = frozenset(
        path for path in application_paths() if route_segments(path) not in addressed
    )

    assert unnamed == ROUTES_NO_CLIENT_LITERAL_NAMES, (
        f"the routes src/api.ts names no literal for are {sorted(unnamed)}, not "
        f"{sorted(ROUTES_NO_CLIENT_LITERAL_NAMES)}. These three are written here as "
        f"literals rather than imported from backend/tests/test_route_consumers.py, so "
        f"that two gates reading the pair in opposite directions agree by measurement "
        f"instead of by shared code. When this fails, that module's allowlists are the "
        f"other half of the change"
    )


def test_api_ts_is_the_whole_client_surface():
    """Nothing outside ``src/api.ts`` talks to the backend.

    Required here and not in the sibling gate: narrow scope fails *silently* in
    this direction. A second file requesting a dead path is simply never read,
    where in the mirror direction it would only produce a false alarm.

    Two suffixes are test code by this repository's own convention and are
    excluded: ``*.test.ts``, which ``vitest.config.ts`` collects as the suite,
    and ``*.test-helper.ts``, which only a ``*.test.ts`` imports and which
    therefore reaches no bundle a user runs.
    """
    sources = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(SRC_DIR.rglob("*.ts"))
        if not path.name.endswith((".test.ts", ".test-helper.ts"))
    }

    assert sources, f"no non-test TypeScript source was found under {SRC_DIR}"

    requesting = sorted(name for name, text in sources.items() if "fetch(" in text)
    assert requesting == ["src/api.ts"], (
        f"these files call fetch(): {requesting}. This gate only parses src/api.ts, so a "
        f"request made anywhere else is never checked against the route set at all"
    )

    naming_base_url = sorted(name for name, text in sources.items() if "BACKEND_BASE_URL" in text)
    assert naming_base_url == ["src/api.ts", "src/contracts.ts"], (
        f"these files name BACKEND_BASE_URL: {naming_base_url}. src/contracts.ts declares "
        f"it and src/api.ts uses it; a third file is composing a backend URL somewhere "
        f"this gate cannot see"
    )

    hardcoding_host = sorted(name for name, text in sources.items() if "127.0.0.1" in text)
    assert hardcoding_host == ["src/contracts.ts"], (
        f"these files write the backend host literally: {hardcoding_host}. A request "
        f"built from a hardcoded URL instead of BACKEND_BASE_URL escapes both this gate "
        f"and the one constant that defines where the backend lives"
    )


SYNTHETIC_CLIENT = "\n".join(
    [
        'const a = request("GET", "/plain");',
        "const b = request(`GET`, `/query?limit=${limit}`);",
        "const c = request(`DELETE`, `/thing/${id}`);",
        'const d = request("GET", `/${host}ealth`);',
        'const e = request("GET", "/thing/42");',
        'const f = request("GET", "/plain/");',
        "/** Prose naming /block-comment as an endpoint. */",
        '// const g = request("GET", "/line-comment");',
        'const h = "/history is where entries live";',
        'const i = request("GET", "/stt/" + "mode");',
        "const j = fetch(`${base}/leading-interpolation`);",
    ]
)


def synthetic_literal_texts() -> tuple[str, ...]:
    return tuple(literal.text for literal in synthetic_paths().literals)


def synthetic_matches(route: str) -> bool:
    shape = route_segments(route)
    return any(client_segments(text) == shape for text in synthetic_literal_texts())


def test_a_static_path_written_whole_is_read():
    assert synthetic_matches("/plain")


def test_a_query_string_is_truncated():
    assert synthetic_matches("/query"), (
        "`/query?limit=${limit}` did not address `/query`, so every paginated read in the "
        "client would be reported as a request to a route that does not exist"
    )


def test_a_parameterised_route_is_addressed_by_an_interpolated_segment():
    assert synthetic_matches("/thing/{thing_id}")


def test_an_interpolation_in_the_wrong_position_addresses_nothing():
    assert not synthetic_matches("/health"), (
        "`/${host}ealth` was read as addressing `/health`. An interpolation is a whole "
        "segment or it is nothing -- reading it as a wildcard inside a segment would "
        "certify any route whose text happens to end the same way"
    )


def test_a_concrete_segment_does_not_address_a_parameterised_route():
    assert "/thing/42" in synthetic_literal_texts()
    assert client_segments("/thing/42") != route_segments("/thing/{thing_id}"), (
        "`/thing/42` was read as addressing `/thing/{thing_id}`. A parameterised path is "
        "matched in position rather than truncated at its placeholder, which is what "
        "keeps one interpolated literal from certifying every sibling under the prefix"
    )


def test_a_trailing_slash_does_not_address_the_route_without_one():
    assert "/plain/" in synthetic_literal_texts()
    assert client_segments("/plain/") != route_segments("/plain"), (
        "`/plain/` was read as addressing `/plain`. FastAPI serves the two differently, "
        "so a literal that grew a trailing slash is a client asking for a path the "
        "backend does not serve"
    )


def test_a_path_written_inside_a_comment_is_not_read():
    texts = synthetic_literal_texts()

    assert "/block-comment" not in texts and "/line-comment" not in texts, (
        f"a path written inside a comment was read as a request: {texts}. This is the one "
        f"property that separates a parse from every pattern over the source text, and it "
        f"is why the first attempt at this check was withdrawn: src/api.ts names paths in "
        f"its own prose, so a text rule reports requests the program never makes"
    )


def test_a_prose_string_that_begins_like_a_path_is_not_read():
    assert "/history is where entries live" not in synthetic_literal_texts(), (
        "a sentence beginning with a path was read as a request path. The whitespace "
        "clause is what excludes it, and it is the only heuristic in this module"
    )


def test_a_concatenated_path_is_reported_assembled():
    shapes = [(item.text, item.shape) for item in synthetic_paths().assembled]

    assert ("/stt/", "concatenation with +") in shapes, (
        f'`"/stt/" + "mode"` was not reported as assembled: {shapes}. Neither half of '
        f"a concatenation is the path being requested, so without this finding the check "
        f"would compare `/stt/` against the route set and report the wrong thing"
    )


def test_a_leading_interpolation_template_is_reported_assembled():
    shapes = [(item.text, item.shape) for item in synthetic_paths().assembled]

    assert ("/leading-interpolation", "template with a leading interpolation") in shapes, (
        f"`${{base}}/leading-interpolation` was not reported as assembled: {shapes}. The "
        f"reconstruction rule cannot see a path in that position at all -- the literal "
        f"list is empty of it -- so this finding is the only thing standing between a "
        f"request written that way and a silent pass"
    )
