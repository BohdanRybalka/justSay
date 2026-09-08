"""Seven values exist in two or three languages at once; the copies must agree.

Each value has exactly one nominated declaration per language, and this module
reads every declaration as **text** so it needs no TypeScript compiler, no Rust
toolchain and no TOML parser -- the same shape ``test_version_consistency.py``
uses for the three version manifests (ADR 030). Nothing here imports ``app``:
``app.audio.__init__`` pulls fastapi and both recorders, and the ``backend-lint``
CI job installs no audio extra, so an import would pass locally and fail there.

Two mechanisms per value where the shape allows it. A **declared-sites table**
maps each path to one extractor and catches a site that went stale. An **orphan
scan** over a bounded, enumerated file set catches a site nobody added to the
table. The scan set is enumerated rather than globbed from the repository root
because ``backend/build/`` is gitignored and holds a stale copy of
``app/core/config.py``: an unbounded walk would fail on any machine that has run
``pip install -e`` and pass in CI.

The seven are the backend port, the masked-key sentinel, the upload allowlist
and its cap, the Tauri event names, the session-id alphabet, the Tauri command
names, and the two application data directory names.

ADR 045 records why these values are pinned rather than generated. Its
amendment nominates Rust as the canonical declaration for the Tauri command
names, which have exactly two parties and one definer.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PY = REPO_ROOT / "backend" / "app" / "core" / "config.py"
CONSTANTS_PY = REPO_ROOT / "backend" / "app" / "core" / "constants.py"
AUDIO_FORMATS_PY = REPO_ROOT / "backend" / "app" / "core" / "audio_formats.py"
APP_PATHS_PY = REPO_ROOT / "backend" / "app" / "core" / "app_paths.py"
SESSION_PY = REPO_ROOT / "backend" / "app" / "audio" / "session.py"
CONTRACTS_TS = REPO_ROOT / "src" / "contracts.ts"
LIB_RS = REPO_ROOT / "src-tauri" / "src" / "lib.rs"
BACKEND_RS = REPO_ROOT / "src-tauri" / "src" / "backend.rs"

RUST_SOURCE_DIR = REPO_ROOT / "src-tauri" / "src"

_PORT_SITES: dict[Path, str] = {
    CONFIG_PY: r"^    port: int = (\d+)$",
    CONTRACTS_TS: r"^export const BACKEND_PORT = (\d+);$",
    BACKEND_RS: r"^pub const PORT: u16 = (\d+);$",
    REPO_ROOT / "src-tauri" / "tauri.conf.json": (
        r'connect-src [^";]*http://127\.0\.0\.1:(\d+) http://localhost:(\d+)'
    ),
    REPO_ROOT / "package.json": r"--port (\d+)",
    REPO_ROOT / "backend" / ".env.example": r"^JUSTSAY_PORT=(\d+)$",
    REPO_ROOT / "backend" / "scripts" / "smoke_sidecar.py": r'"--port", type=int, default=(\d+)',
    REPO_ROOT / "backend" / "build_sidecar.spec": r"--port (\d+)",
    REPO_ROOT / ".github" / "workflows" / "release.yml": r"--port (\d+)",
}

_PORT_SCAN_PATTERNS: tuple[str, ...] = (
    r"127\.0\.0\.1:(\d+)",
    r"localhost:(\d+)",
    r"port[ =:]+(\d{4,5})",
    r"u(?:8|16|32|64)\s*=\s*(\d{4,5})",
)

_NON_BACKEND_PORTS: dict[str, str] = {
    "5173": "Vite dev server -- vite.config.ts, the tauri.conf.json devUrl, and the CORS origins",
    "11434": "Ollama's default host, written out at every provider and settings default",
    "8878": "the local whisper.cpp server in app/stt/local_whisper_cpp.py, single-language",
}

_SCAN_GLOBS: tuple[tuple[str, str], ...] = (
    ("src", "**/*.ts"),
    ("src-tauri/src", "**/*.rs"),
    ("src-tauri", "*.json"),
    ("backend/app", "**/*.py"),
    ("backend/scripts", "*.py"),
    ("backend/tests", "*.py"),
    ("backend", ".env.example"),
    ("backend", "build_sidecar.spec"),
    (".", "package.json"),
    (".", "vite.config.ts"),
    (".github/workflows", "*.yml"),
)

_TYPESCRIPT_EVENT_PATTERNS: tuple[str, ...] = (
    r'\b(?:emit|listen|once)\s*(?:<[^>]*>)?\s*\(\s*"([^"]+)"',
    r'\b(?:emitTo|listenTo|onceTo)\s*(?:<[^>]*>)?\s*\(\s*"[^"]*"\s*,\s*"([^"]+)"',
)

_RUST_EMIT_PATTERNS: tuple[str, ...] = (
    r'\.emit\(\s*"([^"]+)"',
    r'\.emit_to\(\s*"[^"]*"\s*,\s*"([^"]+)"',
)

_TYPESCRIPT_EMITTER_TEMPLATES: tuple[str, ...] = (
    r"\bemit\s*(?:<[^>]*>)?\s*\(\s*{constant}\b",
    r"\bemitTo\s*(?:<[^>]*>)?\s*\(\s*\"[^\"]*\"\s*,\s*{constant}\b",
)

_TYPESCRIPT_INVOKE_PATTERN = "\\binvoke\\w*\\s*(?:<[^>]*>)?\\s*\\(\\s*[\"']([^\"']+)[\"']"

_RUST_COMMAND_PATTERN = (
    r"#\[tauri::command(?:\([^)]*\))?\]"
    r"(?:\s*(?://[^\n]*|#\[[^\]]*\]))*"
    r"\s*(?:pub\s*(?:\([^)]*\)\s*)?)?(?:async\s+)?fn\s+(\w+)"
)

_RUST_HANDLER_PATTERN = r"tauri::generate_handler!\[(.*?)\]"

_RUST_HANDLER_ENTRY_PATTERN = r"(?:[A-Za-z_]\w*\s*::\s*)*([A-Za-z_]\w*)"

_RUST_DATA_DIR_PATTERN = r'if\s+\w+\s*\{\s*"([^"]*)"\s*\}\s*else\s*\{\s*"([^"]*)"\s*\}'

_TYPESCRIPT_COMMENT_OR_STRING_PATTERN = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|`(?:\\.|[^`\\])*`" r"|/\*.*?\*/" r"|//[^\n]*",
    re.DOTALL,
)

_QUOTED_SPAN_PATTERN = re.compile("\"[^\"\n]*\"|'[^'\n]*'")

_SHARED_MODEL_CACHE_ALLOWLIST: dict[tuple[str, str], tuple[int, str]] = {
    (
        "backend/app/stt/local_whisper_cpp_cmd.py",
        'return Path.home() / ".justsay" / "models" / "whisper-cpp"'
        ' / f"ggml-{model_size}.bin"',
    ): (
        1,
        "the downloaded GGML model cache is deliberately shared between dev and "
        "production -- docs/adr/012-dev-mode-data-directory-isolation.md, pinned by "
        "test_model_cache_stays_shared_between_dev_and_production",
    ),
}


@functools.cache
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract(path: Path, pattern: str) -> list[str]:
    matches = re.findall(pattern, _read(path), re.MULTILINE)
    assert matches, f"no declaration matching {pattern!r} found in {path}"
    values: list[str] = []
    for match in matches:
        values.extend([match] if isinstance(match, str) else match)
    return values


def _extract_groups(path: Path, pattern: str) -> list[tuple[str, ...]]:
    """Every match of ``pattern`` as its own tuple of groups.

    ``_extract`` flattens the groups of every match into one list, which is
    what a single-value declaration wants. A declaration carrying two values
    at once needs them kept together, so that a second declaration in the same
    file is reported as a disagreement rather than unpacked into a crash.
    """
    matches = re.findall(pattern, _read(path), re.MULTILINE)
    assert matches, f"no declaration matching {pattern!r} found in {path}"
    return [match if isinstance(match, tuple) else (match,) for match in matches]


@functools.cache
def _scanned_files() -> tuple[Path, ...]:
    found: list[Path] = []
    for directory, glob in _SCAN_GLOBS:
        found.extend(sorted((REPO_ROOT / directory).glob(glob)))
    assert found, "the enumerated scan set matched no files at all"
    return tuple(found)


def _whole_quoted_string_matcher(names: tuple[str, ...]) -> Callable[[str], bool]:
    """A value is written out only when it is the entire quoted string.

    What the masked-key sentinel needs: ``'***'`` is a declaration and the
    same three characters inside a longer literal are not, so the closing
    quote has to follow the value immediately.
    """
    quoted = re.compile("|".join("[\"']" + re.escape(name) + "[\"']" for name in names))
    return lambda line: quoted.search(line) is not None


def _path_segment_matcher(names: tuple[str, ...]) -> Callable[[str], bool]:
    """A value is written out when it is a path segment inside a quoted string.

    What a directory name needs, and what the whole-string rule above cannot
    express: ``"~/.justsay/history.db"`` writes the production directory name
    down just as surely as ``".justsay"`` does, and it is the likelier drift
    form of the two. Segment boundaries are a quote or a slash, which is what
    keeps ``.justsay`` from matching inside ``.justsay-dev``, inside the
    write-probe prefix ``f".justsay-write-probe-{...}"``, and inside the bundle
    identifier ``"com.justsay.app"``.

    Only text inside a quoted span counts, so the many prose mentions of
    ``~/.justsay/logs`` in docstrings and Rust line comments stay silent: a
    docstring line carries no quote pair of its own.
    """
    segment = re.compile("|".join(rf"(?:^|/){re.escape(name)}(?:/|$)" for name in names))

    def matches(line: str) -> bool:
        return any(segment.search(span[1:-1]) for span in _QUOTED_SPAN_PATTERN.findall(line))

    return matches


def _quoted_literal_strays(
    matches: Callable[[str], bool],
    declaring: tuple[Path, ...],
    allowlist: dict[tuple[str, str], tuple[int, str]],
) -> tuple[list[str], list[str]]:
    """Sites outside the declaring files where ``matches`` says a value is written.

    Shared by the masked-key sentinel and the data directory names: one walk
    over the enumerated scan set, skipping the declaring files and test files.
    The two callers need different matching rules -- a sentinel is the whole
    quoted string, a directory name is one segment of a path -- so the rule is
    passed in rather than one of them being weakened to fit the other.

    The allowlist is keyed by ``(path, exact stripped source line)`` and its
    value carries the number of occurrences that line is expected to have.
    Keying on the text alone exempted every byte-identical copy of it, which is
    exactly how a second stray appears; counting exempts that many and reports
    the rest. An entry whose line has changed, gone, or become rarer comes back
    in the second list as stale rather than exempting that file forever with
    nobody noticing.
    """
    strays: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    for path in _scanned_files():
        if path in declaring or path.name.startswith("test_"):
            continue
        if path.name.endswith((".test.ts", "_test.py")):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            if not matches(line):
                continue
            site = (rel, line.strip())
            seen[site] = seen.get(site, 0) + 1
            if site in allowlist and seen[site] <= allowlist[site][0]:
                continue
            strays.append(f"{rel}:{lineno}")
    stale = [
        f"{rel} carries the line {line!r} {seen.get((rel, line), 0)} times, not {expected}"
        for (rel, line), (expected, _) in allowlist.items()
        if seen.get((rel, line), 0) < expected
    ]
    return strays, stale


def _canonical_port() -> str:
    return _extract(CONFIG_PY, _PORT_SITES[CONFIG_PY])[0]


def test_the_backend_port_is_the_same_number_everywhere() -> None:
    """Every nominated declaration of the backend port carries one number.

    Mutation-checked: changing the default in app/core/config.py and nothing
    else fails this test, and the message names all nine declaring files.
    """
    declared = {path: _extract(path, pattern) for path, pattern in _PORT_SITES.items()}
    canonical = declared[CONFIG_PY][0]
    disagreeing = {
        path: values for path, values in declared.items() if set(values) != {canonical}
    }
    assert not disagreeing, (
        f"{CONFIG_PY} declares the backend port as {canonical}, but the declared sites "
        "disagree. Every one of these must carry the same number: "
        + "; ".join(
            f"{path.relative_to(REPO_ROOT).as_posix()} declares {', '.join(values)}"
            for path, values in declared.items()
        )
    )


def test_no_undeclared_port_literal_exists() -> None:
    """No file in the scanned set writes a loopback number the table does not own.

    Rust declares its port as ``pub const PORT: u16 = 9377;``, which the
    ``port[ =:]+`` pattern cannot reach because ``[ =:]`` does not match ``u``,
    so an unsigned-integer assignment is scanned as well. That pattern accepts
    any of Rust's unsigned widths and tolerates missing spaces, because
    ``cargo fmt`` runs in no CI job here and so nothing normalises either.

    Mutation-checked three times: appending a stray four-digit loopback literal
    to a scanned file fails this test naming that file and line; appending a
    second ``pub const ...: u16 = <a different number>;`` to
    src-tauri/src/backend.rs fails it the same way; and so does the same
    declaration written ``u32`` with no spaces around the equals sign. The three
    allowlisted non-backend numbers stay green through all of them.

    This test scans its own file, so no non-canonical port number may be written
    out here -- not in an assertion message and not in this docstring. The
    failure is loud and names the line; ADR 045 and the spec's ``## Risks``
    record why that is accepted rather than excluded from the scan set.
    """
    canonical = _canonical_port()
    allowed = {canonical} | set(_NON_BACKEND_PORTS)
    orphans: list[str] = []
    for path in _scanned_files():
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            for pattern in _PORT_SCAN_PATTERNS:
                for value in re.findall(pattern, line, re.IGNORECASE):
                    if value not in allowed:
                        rel = path.relative_to(REPO_ROOT).as_posix()
                        orphans.append(f"{rel}:{lineno} carries {value}")
    assert not orphans, (
        f"undeclared loopback numbers found (the backend uses {canonical}; allowlisted "
        f"elsewhere: {', '.join(f'{n} = {why}' for n, why in _NON_BACKEND_PORTS.items())}): "
        + "; ".join(orphans)
    )


def test_the_masked_key_sentinel_agrees_across_languages() -> None:
    """The sentinel the backend returns for a stored cloud key exists twice.

    The stray scan matches either quote character. Nothing in this repository
    enforces double quotes -- backend/pyproject.toml selects no ruff ``Q`` rule
    and ``ruff format`` runs in no npm script and no CI step -- so a Python
    ``'***'`` would otherwise pass a double-quote-only scan silently.

    Mutation-checked twice: changing MASKED_API_KEY in app/core/constants.py
    fails this test naming src/contracts.ts, and planting a single-quoted
    sentinel in a non-test file under backend/app/ fails it naming that line.
    """
    python_value = _extract(CONSTANTS_PY, r'^MASKED_API_KEY: str = "([^"]*)"$')[0]
    typescript_value = _extract(CONTRACTS_TS, r'^export const MASKED_API_KEY = "([^"]*)";$')[0]
    assert python_value == typescript_value, (
        f"{CONSTANTS_PY} declares the masked-key sentinel as {python_value!r} but "
        f"{CONTRACTS_TS} declares it as {typescript_value!r}"
    )

    strays, _ = _quoted_literal_strays(
        _whole_quoted_string_matcher((python_value,)), (CONSTANTS_PY, CONTRACTS_TS), {}
    )
    assert not strays, (
        f"the masked-key sentinel {python_value!r} is written out, in either quote style, "
        "away from its two declarations "
        f"({CONSTANTS_PY.name}, {CONTRACTS_TS.name}); import it instead: " + ", ".join(strays)
    )


def _product(expression: str) -> int:
    result = 1
    for factor in expression.split("*"):
        result *= int(factor.strip())
    return result


def _extensions_in(path: Path, opening: str, closing: str) -> set[str]:
    text = _read(path)
    start = text.index(opening) + len(opening)
    block = text[start : start + text[start:].index(closing)]
    found = set(re.findall(r'"(\.[a-z0-9]+)"', block))
    assert found, f"no audio extensions found between {opening!r} and {closing!r} in {path}"
    return found


def test_the_upload_allowlist_and_cap_agree_across_languages() -> None:
    """The accepted audio extensions and the upload cap exist in both languages.

    Mutation-checked: deleting one entry from MIME_BY_AUDIO_EXTENSION fails this
    test naming that extension and both sides.
    """
    python_extensions = _extensions_in(
        AUDIO_FORMATS_PY, "MIME_BY_AUDIO_EXTENSION: dict[str, str] = {", "}"
    )
    typescript_extensions = _extensions_in(
        CONTRACTS_TS, "ACCEPTED_AUDIO_EXTENSIONS: readonly string[] = [", "]"
    )
    assert python_extensions == typescript_extensions, (
        f"the accepted audio extensions disagree: only in {AUDIO_FORMATS_PY.name}: "
        f"{sorted(python_extensions - typescript_extensions)}; only in "
        f"{CONTRACTS_TS.name}: {sorted(typescript_extensions - python_extensions)}"
    )

    python_cap = _product(_extract(CONSTANTS_PY, r"^MAX_UPLOAD_SIZE: int = ([\d *]+)$")[0])
    typescript_cap = _product(
        _extract(CONTRACTS_TS, r"^export const MAX_UPLOAD_BYTES = ([\d *]+);$")[0]
    )
    assert python_cap == typescript_cap, (
        f"{CONSTANTS_PY.name} caps uploads at {python_cap} bytes but "
        f"{CONTRACTS_TS.name} caps them at {typescript_cap}"
    )


def _typescript_source_files() -> list[Path]:
    return [
        path
        for path in sorted((REPO_ROOT / "src").glob("**/*.ts"))
        if not path.name.endswith(".test.ts") and path != CONTRACTS_TS
    ]


@functools.cache
def _typescript_code(path: Path) -> str:
    """The file's text with every ``//`` and ``/* */`` comment blanked out.

    A comment is not a call site and not a declaration, but a text scan cannot
    tell the difference: this repository is full of JSDoc that quotes the very
    identifiers and command names these pins extract, so a JSDoc line mentioning
    ``invoke("widget_ready")`` made a deleted call look present, and a ``//``
    line carrying a misspelt name was reported as a real invocation. Both
    directions are mutation-checked in
    ``test_every_tauri_command_name_is_defined_and_registered``.

    Comment bodies are replaced by spaces rather than removed, so every line
    number and every column stays where it was. String literals are matched
    first and passed through untouched, so a ``//`` inside a quoted URL is not
    mistaken for the start of a comment.
    """

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("/"):
            return re.sub(r"[^\n]", " ", token)
        return token

    return _TYPESCRIPT_COMMENT_OR_STRING_PATTERN.sub(blank, _read(path))


def _typescript_source_lines() -> list[tuple[str, int, str]]:
    lines: list[tuple[str, int, str]] = []
    for path in _typescript_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(_typescript_code(path).splitlines(), start=1):
            lines.append((rel, lineno, line))
    return lines


def test_every_tauri_event_name_is_declared_once() -> None:
    """Every Tauri event name is declared in contracts.ts, emitted and heard.

    Two assertions, because a name can go wrong in two directions and neither
    raises at runtime -- Tauri drops an event nobody listens for and never fires
    a listener nobody emits to.

    *used is a subset of declared*: no emit(), listen() or once() site under
    src/, no window-targeted emitTo() there, and no .emit() or .emit_to() under
    src-tauri/src/, passes a bare string the module does not declare. The
    targeted forms take the event name as their second argument, so they need
    their own patterns -- without them a wrong name reached the event bus
    unnoticed. Mutation-checked: renaming the value of EVENT_MEETING_TOGGLE
    fails this test naming src-tauri/src/lib.rs; a bare string literal passed to
    emit() in a non-test file under src/ fails it naming that file; and so does
    an .emit_to() carrying a misspelt name while the existing .emit() stays.

    *declared is a subset of used*: every EVENT_* the module declares has at
    least one emitter and at least one listener, counted over TypeScript
    identifier uses plus the Rust literal emit sites. It does **not** assert
    that the two sit on opposite sides of any boundary -- a name emitted and
    listened for within one file satisfies it. Mutation-checked: deleting
    the sole emit(EVENT_SETTINGS_CHANGED) call in settings/tabs/general.ts, with
    the constant and its listener left in place, fails this test naming
    EVENT_SETTINGS_CHANGED as declared and listened for but never emitted.
    """
    declared_pairs = re.findall(
        r'^export const (EVENT_[A-Z_]+) = "([^"]+)";$', _read(CONTRACTS_TS), re.MULTILINE
    )
    assert declared_pairs, f"no EVENT_* declaration found in {CONTRACTS_TS}"
    declared: dict[str, str] = dict(declared_pairs)
    values = set(declared.values())

    typescript_lines = _typescript_source_lines()

    used: list[tuple[str, int, str]] = []
    for rel, lineno, line in typescript_lines:
        for pattern in _TYPESCRIPT_EVENT_PATTERNS:
            for name in re.findall(pattern, line):
                used.append((rel, lineno, name))
    rust_emits: list[tuple[str, int, str]] = []
    for path in _rust_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            for pattern in _RUST_EMIT_PATTERNS:
                for name in re.findall(pattern, line):
                    rust_emits.append((rel, lineno, name))
    used.extend(rust_emits)

    assert rust_emits, (
        f"{LIB_RS.relative_to(REPO_ROOT).as_posix()} no longer emits any event by name; the "
        "Rust half of this contract has moved and the extractor must move with it"
    )

    undeclared = [
        f"{rel}:{lineno} uses {name!r}" for rel, lineno, name in used if name not in values
    ]
    assert not undeclared, (
        f"{CONTRACTS_TS.name} declares the event names {sorted(values)}; these sites use a "
        "name it does not declare, so an emitter and its listener can drift apart silently: "
        + "; ".join(undeclared)
    )

    unpaired: list[str] = []
    for constant, value in declared.items():
        emitters = [
            f"{rel}:{lineno}"
            for rel, lineno, line in typescript_lines
            if any(
                re.search(template.format(constant=constant), line)
                for template in _TYPESCRIPT_EMITTER_TEMPLATES
            )
        ]
        emitters += [f"{rel}:{lineno}" for rel, lineno, name in rust_emits if name == value]
        listeners = [
            f"{rel}:{lineno}"
            for rel, lineno, line in typescript_lines
            if re.search(rf"\b(?:listen|once)\s*(?:<[^>]*>)?\s*\(\s*{constant}\b", line)
        ]
        if not emitters or not listeners:
            unpaired.append(
                f"{constant} ({value!r}) has "
                f"{'no emitter' if not emitters else 'emitters ' + ', '.join(emitters)} and "
                f"{'no listener' if not listeners else 'listeners ' + ', '.join(listeners)}"
            )
    assert not unpaired, (
        f"{CONTRACTS_TS.name} declares an event name that is not a two-party contract; every "
        "declared name needs at least one emitter and at least one listener, or it is a dead "
        "string and does not belong in this module: " + "; ".join(unpaired)
    )


def test_the_session_id_alphabet_agrees_across_languages() -> None:
    """The regex a client mints against and the one the backend validates with.

    A drift here is not a wrong id, it is every id wrong: the client mints to
    its own spelling and the backend answers 422 to all of them, so dictation
    stops working entirely on a one-character edit that nothing else notices.
    The TypeScript side is a regex literal and the Python side a string, so the
    comparison is of the pattern text between the delimiters.

    Mutation-checked: widening the Python pattern to ``[0-9a-fA-F]`` and
    leaving src/contracts.ts alone fails this test printing both spellings.
    """
    python_pattern = _extract(SESSION_PY, r'^SESSION_ID_PATTERN = "([^"]*)"$')[0]
    typescript_pattern = _extract(CONTRACTS_TS, r"^export const SESSION_ID_PATTERN = /(.*)/;$")[0]
    assert python_pattern == typescript_pattern, (
        f"{SESSION_PY.name} validates session ids against {python_pattern!r} but "
        f"{CONTRACTS_TS.name} mints them against {typescript_pattern!r}"
    )


def _rust_source_files() -> list[Path]:
    """Every Rust source file under src-tauri/src/, submodules included.

    The walk is recursive because ``src-tauri/src/commands/mod.rs`` is an
    ordinary place for a Tauri command to live and
    ``tauri::generate_handler![commands::widget_ready]`` registers it from
    there. A non-recursive glob made the two halves of this pin disagree about
    whether submodules exist: the registration parser accepted the qualified
    path while the definition extractor never saw the function, so a correct
    TypeScript caller was reported as invoking an undefined name.
    """
    return sorted(RUST_SOURCE_DIR.glob("**/*.rs"))


def _rust_command_definitions() -> dict[str, list[str]]:
    """Every ``#[tauri::command]`` function under src-tauri/src/, and its files.

    The whole directory is walked rather than lib.rs alone, the way the event
    pin's Rust half already does: a command defined in backend.rs or main.rs is
    as real as one defined in lib.rs, and reading only lib.rs would report its
    correct TypeScript caller as invoking an undefined name.

    Every definer of a name is kept, not the last one seen. Two modules may
    each define a ``#[tauri::command] fn widget_ready``; overwriting made every
    assertion message name whichever file sorted later, which is the one place
    a reader of that message would not look.

    The attribute accepts arguments (``#[tauri::command(rename_all = ...)]``),
    any run of doc-comment or attribute lines may sit between it and the
    function, and every visibility form is accepted. Each of those is legal,
    ordinary Rust -- ``///`` doc comments are what CLAUDE.md asks for -- and
    each made this extractor miss the function and blame correct TypeScript.
    """
    defined: dict[str, list[str]] = {}
    for path in _rust_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for name in re.findall(_RUST_COMMAND_PATTERN, _read(path)):
            defined.setdefault(name, []).append(rel)
    assert defined, "no #[tauri::command] function was found under src-tauri/src/"
    return defined


def _rust_registered_commands() -> set[str]:
    """Every name listed in a ``tauri::generate_handler!`` macro, unqualified.

    All occurrences of the macro are read, not the first: a second builder --
    a mobile entry point, a second window -- registers its own commands, and
    stopping at the first would report every one of them as unregistered.

    The body is split on commas and each entry must be a path in full, so only
    a real registration contributes a name. Harvesting identifiers from the
    body instead let anything written inside the macro through: a
    ``#[cfg(desktop)]`` gate contributed ``cfg`` and ``desktop``, and since
    unregistered names are what is left after subtracting this set, a stray
    entry can only ever hide a command that is genuinely unreachable. An entry
    this parser does not recognise contributes nothing, which fails loudly
    rather than passing silently. Each entry is reduced to its last path
    segment, because ``tauri::generate_handler![commands::widget_ready]`` is
    legal and registers the function named ``widget_ready``.
    """
    registered: set[str] = set()
    for path in _rust_source_files():
        for block in re.findall(_RUST_HANDLER_PATTERN, _read(path), re.DOTALL):
            for entry in block.split(","):
                match = re.fullmatch(_RUST_HANDLER_ENTRY_PATTERN, entry.strip())
                if match:
                    registered.add(match.group(1))
    assert registered, "no tauri::generate_handler! macro was found under src-tauri/src/"
    return registered


def _typescript_invoke_sites() -> list[tuple[str, int, str]]:
    """Every literal command name passed to an ``invoke``-prefixed call.

    The scan is over the whole file text rather than line by line, and the line
    number comes from the match offset. A call reformatted across several lines
    -- which prettier, an editor or a hand edit can produce at any time and
    nothing here forbids -- would otherwise read as no call at all, and the
    test would name a working caller as missing.
    """
    sites: list[tuple[str, int, str]] = []
    for path in _typescript_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = _typescript_code(path)
        for match in re.finditer(_TYPESCRIPT_INVOKE_PATTERN, text):
            sites.append((rel, text.count("\n", 0, match.start()) + 1, match.group(1)))
    return sites


def test_every_tauri_command_name_is_defined_and_registered() -> None:
    """Every Tauri command name is defined in Rust, called, and registered.

    Rust is the canonical declaration here, unlike the event names: a
    ``#[tauri::command] fn`` *is* the definition and there are exactly two
    parties, one of which unambiguously defines. ADR 045's amendment records
    that carve-out.

    Three assertions, because a command name goes wrong in three ways and none
    of them raises at build time -- a misspelt call reaches no command, an
    unreferenced command is dead, and an unregistered command is unreachable at
    runtime with nothing said anywhere.

    The invoke pattern accepts trailing word characters, which is what reaches
    the local ``invokeShell`` wrapper in src/widget/widget.ts, and an optional
    generic parameter list, which src/api.ts needs. Both quote styles are
    accepted for the reason the masked-key test already records: no eslint,
    prettier or biome is installed, so nothing normalises quotes in TypeScript.

    Comments are blanked out of the TypeScript text before it is scanned, in
    both directions. This repository's JSDoc quotes command names freely, so a
    doc line mentioning a call satisfied the caller check for a call that had
    been deleted, and a ``//`` line carrying a misspelt name was reported as a
    site invoking an undefined command.

    The known gap, accepted rather than closed: a *misspelt duplicate* call
    through a wrapper whose name does not start with ``invoke``, to a command
    that already has a correct call site elsewhere, passes all three.

    Mutation-checked fourteen times, each applied alone. Misspelling the
    widget-ready command name at src/widget/widget.ts fails this test naming
    that file and line; deleting the settings-window call fails it naming the
    command as defined but never called; removing a name from
    ``generate_handler!`` fails it naming that command as unregistered;
    renaming ``invokeShell`` throughout widget.ts fails it naming the three
    commands that wrapper reaches. Planting a ``#[tauri::command]`` function in
    src-tauri/src/backend.rs fails it naming backend.rs as the definer of an
    uncalled command, where reading lib.rs alone would have missed it. A second
    ``tauri::generate_handler!`` macro carrying the four names keeps it green,
    where reading only the first occurrence of the macro made the four read as
    unregistered. Writing one registration path-qualified as
    ``commands::widget_ready`` keeps it green, where splitting the macro body
    on commas alone made that correct registration fail. Reformatting the
    src/api.ts call across three lines keeps it green, where the line-by-line
    scan reported its command as never invoked.

    Six more, added after the second review pass. Deleting the real
    ``invokeShell("widget_ready")`` call while adding a JSDoc line that
    mentions it fails this test naming the command as defined but never called,
    where the comment alone kept it green. A ``//`` line carrying a misspelt
    command name keeps it green, where it was reported as
    ``src/api.ts:234 invokes 'get_backend_tokn'``. Writing the attribute as
    ``#[tauri::command(rename_all = "snake_case")]``, putting a ``///`` doc
    comment between the attribute and ``fn``, and writing ``pub(crate) fn``
    each keep it green, where each made the extractor miss the function and
    fail naming a correct TypeScript line. Moving a command into
    ``src-tauri/src/commands/mod.rs`` and registering it as
    ``commands::get_backend_token`` keeps it green, where the non-recursive
    walk missed the definition the registration parser already accepted. And
    replacing a registration entry with a ``#[cfg(<that command name>)]`` gate
    fails this test naming that command as unregistered, where harvesting bare
    identifiers from the macro body read the gate as the registration.
    """
    defined = _rust_command_definitions()
    registered = _rust_registered_commands()
    called = _typescript_invoke_sites()
    assert called, "no invoke() call site with a literal command name was found under src/"

    declaring_files = sorted({rel for files in defined.values() for rel in files})

    undefined = [
        f"{rel}:{lineno} invokes {name!r}" for rel, lineno, name in called if name not in defined
    ]
    assert not undefined, (
        f"{declaring_files} define the commands {sorted(defined)}; these sites invoke a name "
        "none of them defines, so the call reaches no command and fails only at runtime: "
        + "; ".join(undefined)
    )

    invoked = {name for _, _, name in called}
    uncalled = sorted(name for name in defined if name not in invoked)
    assert not uncalled, (
        "these Rust commands are invoked by no TypeScript site by literal name: "
        + "; ".join(f"{name} (defined in {', '.join(defined[name])})" for name in uncalled)
        + ". Either the caller is gone, or a wrapper stopped being recognised by the invoke "
        "pattern this test extracts with"
    )

    unregistered = sorted(name for name in defined if name not in registered)
    assert not unregistered, (
        "tauri::generate_handler! registers none of these commands, so they are unreachable "
        "at runtime with no error anywhere: "
        + "; ".join(f"{name} (defined in {', '.join(defined[name])})" for name in unregistered)
    )


def test_the_app_data_directory_names_agree_across_languages() -> None:
    """Both application data directory names are declared once per language.

    What is comparable across this boundary is the two name strings, not the
    resolution logic: Python resolves the data root from an env override,
    ``sys.frozen`` and a force-dev flag, while Rust picks a name from
    ``cfg!(debug_assertions)`` and uses it for exactly one thing, the sidecar
    log directory. Rust does not decide Python's data root -- it passes the
    force-dev flag and lets Python resolve. The failure this pin prevents is
    the one backend.rs promises out loud: the sidecar log landing in a
    different directory from the sidecar's own history and settings files.

    **Unpinned and named as such**, three ways the two sides can still land in
    different directories at runtime, none of them visible to a text pin:

    - That ``cfg!(debug_assertions)`` and ``sys.frozen`` agree on any given
      launch is a runtime property.
    - With ``JUSTSAY_DATA_DIR`` set, Python resolves that override while Rust
      writes to ``~/<name>/logs/sidecar.log`` regardless, because
      ``append_sidecar_log`` reads only ``USERPROFILE``/``HOME`` and never
      reads ``JUSTSAY_DATA_DIR``.
    - In a release build, a child process inheriting an externally set
      ``JUSTSAY_FORCE_DEV_DATA_DIR=1`` makes Python resolve the development
      name while Rust has already chosen the production one.

    The Rust pattern tolerates arbitrary whitespace because ``cargo fmt`` runs
    in no CI job here. Its matches are kept as pairs, so a second declaration
    in backend.rs is reported as a disagreement instead of unpacked into a
    ValueError naming neither the file nor the values.

    The orphan scan matches the name as a *path segment* inside a quoted
    literal, not as the whole quoted string. A directory name is written down
    just as completely by ``"~/.justsay/history.db"`` as by ``".justsay"``, and
    the longer form is the likelier drift: the whole-string rule this scan was
    first built with left it invisible. Segment boundaries are a quote or a
    slash, which is what keeps the production name from matching inside the
    development name, inside the write probe in user_settings.py, and inside
    the bundle identifier in tauri.conf.json, and requiring a quoted span is
    what keeps the prose mentions of the directory in docstrings and Rust line
    comments silent. The masked-key sentinel keeps the whole-string rule -- a
    sentinel is not a path -- so the shared scan takes the rule as an argument
    rather than one caller's rule being widened to fit the other.

    Test files are skipped following the masked-key precedent --
    test_app_paths.py writes both names out on purpose. The allowlist exempts
    one exact source line *and states how many times it may appear*, so the
    shared model cache stays legal without the file becoming a blind spot:
    keying on the text alone exempted every byte-identical copy of that line,
    which is exactly how a second stray gets written.

    The Rust pattern is anchored on the shape of the two quoted values rather
    than on the ``force_dev_data_dir`` identifier, so renaming that local
    variable does not red the suite.

    Mutation-checked ten times, each applied alone: changing the production
    name in app_paths.py fails this test naming src-tauri/src/backend.rs;
    planting a double-quoted production-directory path literal in a non-test
    file under backend/app/ fails it naming that file and line; the same
    literal single-quoted fails it too; deleting the allowlist entry fails it
    naming backend/app/stt/local_whisper_cpp_cmd.py; a *second* quoted literal
    added to that same allowlisted file fails it naming that new line; editing
    the allowlisted line fails it naming the entry as stale; and a second,
    disagreeing data-directory block in backend.rs fails it printing both
    pairs. Three more, added after the second review pass: a byte-identical
    copy of the allowlisted line, in that same file, fails it naming the new
    line, where keying the exemption on the text alone kept it green;
    ``_LEGACY_HISTORY = "~/.justsay/history.db"`` planted in
    backend/app/preferences/user_settings.py fails it naming that line, where
    the whole-string rule left it invisible; and the write probe
    ``f".justsay-write-probe-{...}"`` in that same file stays unflagged.
    """
    production = _extract(APP_PATHS_PY, r'^PROD_DIR_NAME = "([^"]*)"$')[0]
    development = _extract(APP_PATHS_PY, r'^DEV_DIR_NAME = "([^"]*)"$')[0]
    backend_rel = BACKEND_RS.relative_to(REPO_ROOT).as_posix()

    rust_declarations = set(_extract_groups(BACKEND_RS, _RUST_DATA_DIR_PATTERN))
    assert len(rust_declarations) == 1, (
        f"{backend_rel} declares the data directory name pair more than once and the copies "
        f"disagree: {sorted(rust_declarations)}"
    )
    rust_development, rust_production = next(iter(rust_declarations))

    assert (development, production) == (rust_development, rust_production), (
        f"{APP_PATHS_PY.name} names the data directories "
        f"({development!r} dev, {production!r} production) but "
        f"{backend_rel} names them "
        f"({rust_development!r} dev, {rust_production!r} production), so the sidecar log "
        "would be written beside a different history database than the sidecar's own"
    )

    strays, stale = _quoted_literal_strays(
        _path_segment_matcher((production, development)),
        (APP_PATHS_PY, BACKEND_RS),
        _SHARED_MODEL_CACHE_ALLOWLIST,
    )
    assert not stale, (
        "an allowlisted shared-model-cache site no longer exists, so its entry would exempt "
        "that file forever with nobody noticing; delete the entry or update it to the line "
        "that replaced it: " + "; ".join(stale)
    )
    assert not strays, (
        f"the data directory names ({production!r}, {development!r}) are written out, in "
        f"either quote style, away from their two declarations ({APP_PATHS_PY.name}, "
        f"{BACKEND_RS.name}); call resolve_app_data_root() instead. Allowlisted elsewhere: "
        + "; ".join(
            f"{rel}:{line!r} = {why}"
            for (rel, line), (_, why) in _SHARED_MODEL_CACHE_ALLOWLIST.items()
        )
        + ". Undeclared: "
        + ", ".join(strays)
    )
