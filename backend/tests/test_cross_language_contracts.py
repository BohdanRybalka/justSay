"""Nine values exist in two or three languages at once; the copies must agree.

Each value has exactly one nominated declaration per language, and this module
reads every declaration as **text** so it needs no TypeScript compiler, no Rust
toolchain, no Swift toolchain and no TOML parser -- the same shape
``test_version_consistency.py`` uses for the three version manifests (ADR 030).
Reading rather than compiling is what lets the ninth value be pinned at all:
``macos/JustSayAudioTap`` is built only by the macOS release job, so nothing
here can compile a line of it, and until now nothing compared it to the Python
that reads its output either. Nothing here imports ``app``:
``app.audio.__init__`` pulls fastapi and both recorders, and the ``backend-lint``
CI job installs no audio extra, so an import would pass locally and fail there.

Two mechanisms per value where the shape allows it. A **declared-sites table**
maps each path to one extractor and catches a site that went stale. An **orphan
scan** over a bounded, enumerated file set catches a site nobody added to the
table. The scan set is enumerated rather than globbed from the repository root
because ``backend/build/`` is gitignored and holds a stale copy of the
backend tree: an unbounded walk would fail on any machine that has run
``pip install -e`` and pass in CI.

The nine are the backend port, the masked-key sentinel, the upload allowlist
and its cap, the Tauri event names, the session-id alphabet, the Tauri command
names, the two application data directory names, the capture-incident tokens a
meeting recording can report, and the stdout contract between the macOS audio
tap helper and the Python that reads it.

ADR 045 records why these values are pinned rather than generated. Its
amendment nominates Rust as the canonical declaration for the Tauri command
names, which have exactly two parties and one definer.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PY = REPO_ROOT / "backend" / "app" / "config.py"
CONSTANTS_PY = REPO_ROOT / "backend" / "app" / "core" / "constants.py"
AUDIO_FORMATS_PY = REPO_ROOT / "backend" / "app" / "core" / "audio_formats.py"
APP_PATHS_PY = REPO_ROOT / "backend" / "app" / "core" / "app_paths.py"
SESSION_PY = REPO_ROOT / "backend" / "app" / "audio" / "session.py"
MEETING_RECORDER_PY = REPO_ROOT / "backend" / "app" / "audio" / "meeting_recorder.py"
MACOS_TAP_PY = REPO_ROOT / "backend" / "app" / "audio" / "macos_tap.py"
SYSTEM_SOURCE_PY = REPO_ROOT / "backend" / "app" / "audio" / "system_source.py"
AUDIO_TAP_SWIFT = (
    REPO_ROOT / "macos" / "JustSayAudioTap" / "Sources" / "JustSayAudioTap" / "main.swift"
)
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

_RUST_HANDLER_OPENING = "tauri::generate_handler!["

_RUST_ENTRY_ATTRIBUTE_PATTERN = r"#\[[^\]]*\]"

_RUST_HANDLER_ENTRY_PATTERN = r"(?:[A-Za-z_]\w*\s*::\s*)*([A-Za-z_]\w*)"

_RUST_DATA_DIR_PATTERN = r'if\s+\w+\s*\{\s*"([^"]*)"\s*\}\s*else\s*\{\s*"([^"]*)"\s*\}'

_DOCUMENTED_HEADER_PATTERN = r"stdout: (\{[^}]*\})"

_DOCUMENTED_ARGV_PATTERN = r"justsay-audiotap --.*"

_SWIFT_HEADER_KEY_PATTERN = r'\\"(\w+)\\":'

_SWIFT_HEADER_FORMAT_PATTERN = r'\\"format\\":\\"([A-Za-z0-9]+)\\"'

_PYTHON_HEADER_KEY_PATTERN = r'header(?:\.get\(|\[)"(\w+)"'

_TAP_BLOCK_FRAMES_FLAG = "--block-frames"

_SWIFT_BLOCK_SAMPLES = "blockFrames * channels"

_PYTHON_BLOCK_BYTES = (
    "self._settings.meeting_block_frames * self._channels * SAMPLE_BYTES"
)

_PYTHON_DEINTERLEAVE = "interleaved_buffer_to_mono(chunk, self._channels, SAMPLE_DTYPE)"

_TAP_SAMPLE_SPELLINGS: dict[str, tuple[str, str]] = {
    "f32le": ("private var pending: [Float] = []", "MemoryLayout<Float>.size"),
}

_TAP_PYTHON_SPELLINGS: dict[str, tuple[str, int]] = {
    "f32le": ("<f4", 4),
}

_SWIFT_WHOLE_BLOCK_WRITES: tuple[str, ...] = (
    "pending.count % blockSamples",
    "while offset < bytes.count",
    "offset += written",
)

_SWIFT_CHANNEL_GUARDS: tuple[tuple[str, str], ...] = (
    ("consume", "Int(buffer.mNumberChannels) == channels"),
    ("resolveTapBufferIndex", "Int(format.mChannelsPerFrame) == channels"),
)

_SWIFT_TAP_SERIAL_QUEUE = re.compile(
    r'private let queue = DispatchQueue\(\s*label: "com\.justsay\.audiotap\.io"\s*\)'
)

_SWIFT_IOPROC_ON_THE_TAP_QUEUE = re.compile(
    r"AudioDeviceCreateIOProcIDWithBlock\(\s*&ioProcID\s*,\s*aggregateID\s*,\s*queue\s*\)"
)

_SWIFT_FLUSH_FROM_STOP = re.compile(r"queue\.sync\s*\{\s*flushWholeBlocks\(\)\s*\}")

_SWIFT_STDOUT_WRITERS = ["flushWholeBlocks", "writeHeader"]

_SWIFT_FLUSH_CALLERS = ["consume", "stop"]

_SWIFT_COMMENT_OR_STRING_PATTERN = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|/\*.*?\*/" r"|//[^\n]*",
    re.DOTALL,
)

_NOT_A_NEWLINE = re.compile(r"[^\n]")

_SWIFT_FUNCTION_PATTERN = re.compile(r"^[ \t]*(?:private\s+)?func (\w+)\b", re.MULTILINE)

_RUST_COMMENT_OR_STRING_PATTERN = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|/\*.*?\*/" r"|//[^\n]*",
    re.DOTALL,
)

_RUST_TOP_LEVEL_FUNCTION_PATTERN = re.compile(
    r"^(?:pub\s+(?:\([^)]*\)\s*)?)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE
)

_RUST_SETTINGS_WINDOW_PATTERN = re.compile(r'get_webview_window\(\s*"settings"')

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

    Mutation-checked: changing the default in app/config.py and nothing
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


@functools.cache
def _rust_code(path: Path) -> str:
    """One Rust file with every ``//`` and ``/* */`` comment body blanked out.

    A doc comment sits above its ``fn`` and so inside no function body, so a
    ``.show()`` named in Rust prose would count toward the file total and fail
    the completeness assertion below on a file that is correct. Blanking to
    spaces keeps every line and column. A string literal is matched first, so a
    ``//`` inside one survives; a single-quoted span is not matched at all,
    being a lifetime in Rust far more often than a character literal.
    """

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("/"):
            return _NOT_A_NEWLINE.sub(" ", token)
        return token

    return _RUST_COMMENT_OR_STRING_PATTERN.sub(blank, _read(path))


def _rust_top_level_function_bodies(path: Path) -> dict[str, str]:
    """Each top-level ``fn`` of one Rust file, from its signature to its close.

    Bounded by the first ``}`` in column zero after the signature rather than
    by counting braces, which is the shape the Swift reader in this module
    already uses: nothing here parses Rust, and every brace nested inside a
    top-level function closes further in. A closure body is therefore part of
    the function that holds it, which is what "which function does this call
    sit in" has to mean for the tray menu's handler.
    """
    code = _rust_code(path)
    bodies: dict[str, str] = {}
    for match in _RUST_TOP_LEVEL_FUNCTION_PATTERN.finditer(code):
        rest = code[match.end() :]
        closing = re.search(r"\n\}", rest)
        assert closing, (
            f"{path.relative_to(REPO_ROOT).as_posix()}'s fn {match.group(1)} has no "
            "closing brace in column zero, so this reader cannot say where it ends"
        )
        bodies[match.group(1)] = rest[: closing.start()]
    return bodies


def _assert_one_lib_helper_owns_the_settings_window(
    verb: str, helper: str, event: str, also_allowed: tuple[str, ...]
) -> None:
    """Every ``.<verb>()`` in lib.rs sits in ``helper`` or in ``also_allowed``.

    ``helper`` must resolve the settings window, must not discard what the call
    returns, and must emit ``event``, so a call that failed announces nothing.
    """
    call = re.compile(rf"\.{verb}\(")
    code = _rust_code(LIB_RS)
    bodies = _rust_top_level_function_bodies(LIB_RS)
    rel = LIB_RS.relative_to(REPO_ROOT).as_posix()
    assert bodies, (
        f"{rel} yielded no top-level fn at all; the reader has gone blind and every "
        "assertion below would pass on nothing"
    )

    everywhere = len(call.findall(code))
    assert everywhere, (
        f"{rel} no longer calls .{verb}() on any window; the Rust half of this "
        "contract has moved and the extractor must move with it"
    )

    enclosed = {
        name: len(call.findall(body)) for name, body in bodies.items() if call.search(body)
    }
    assert sum(enclosed.values()) == everywhere, (
        f"{rel} calls .{verb}() {everywhere} times but only {sum(enclosed.values())} of "
        f"them sit inside a top-level fn this reader can name; it found {sorted(enclosed)}"
    )
    assert sorted(enclosed) == sorted([helper, *also_allowed]), (
        f"every path that {verb}s the settings window must go through the one helper that "
        f"emits '{event}', or the page and the window disagree about what is on screen "
        f"with nothing to say why (ADR 089); .{verb}() is called from {sorted(enclosed)}"
    )
    assert _RUST_SETTINGS_WINDOW_PATTERN.search(bodies[helper]), (
        f"{helper} no longer resolves the settings window, so the helper this pin routes "
        f"every {verb} through is acting on something else"
    )
    assert re.search(rf'\.emit\(\s*"{event}"', bodies[helper]), (
        f"{helper} {verb}s the window without announcing it, so settings.ts never learns "
        "what happened to the window it is drawing"
    )
    assert not re.search(rf"let\s+_\s*=\s*window\.{verb}\(", bodies[helper]), (
        f"{helper} discards the result of {verb}(), so a {verb} that failed still emits "
        f"'{event}' and the page acts on a window state that never happened"
    )


def test_every_settings_show_site_announces_it() -> None:
    """Showing the settings window and announcing it are one indivisible step.

    ``settings-shown`` is the only thing that tells the page the window came
    back, and a tab that released its polling on the hide has no other way to
    learn it may start again. A second show path that calls ``show()`` without
    emitting therefore does not fail loudly: the window appears, and the badge
    on it stays frozen at whatever it read before the dismissal. That is the
    defect the single helper exists to prevent, and neither compiler can catch
    it (ADR 089).

    So this pin is structural rather than value-shaped, unlike the rest of the
    module. Every ``.show()`` in ``lib.rs`` is mapped to the top-level function
    holding it, and that set must be exactly the announcing helper plus
    ``widget_ready``, which shows the other window. A show added to the tray
    arm, to ``show_settings_window`` or to a new command fails this test naming
    the function; the completeness assertion covers a ``.show()`` this reader
    could place in no function at all.

    The helper's own ``show()`` result may not be discarded, because an
    announcement a failed show still sends resumes the polling into a window
    the user cannot see.

    Mutation-checked: restoring the window resolution and ``show()`` inline in
    the tray menu arm reports ``run`` in the enclosing set, and deleting the
    ``emit`` from the helper fails the last assertion with every other one
    still passing.
    """
    _assert_one_lib_helper_owns_the_settings_window(
        "show", "show_settings", "settings-shown", ("widget_ready",)
    )


def test_every_settings_hide_site_announces_it() -> None:
    """Hiding the settings window and announcing it are one indivisible step.

    The mirror of the show pin, and the half a page can be hurt by in the other
    direction: a ``settings-hidden`` sent for a hide that failed stops the
    polling on a window the user is still looking at, and a second hide path
    added later that forgets the emit leaves it polling for ever (ADR 089).
    """
    _assert_one_lib_helper_owns_the_settings_window(
        "hide", "hide_settings", "settings-hidden", ()
    )


def _assert_only_one_fn_reaches_the_settings_window(verb: str, helper: str, event: str) -> None:
    """Across src-tauri/src/, only ``helper`` names the settings window and ``.<verb>()``s it.

    Each file's calls are counted twice — in the file and inside the fns this
    reader can name — so one it cannot attribute fails here rather than passing
    unseen.
    """
    call = re.compile(rf"\.{verb}\(")
    reaching: dict[str, str] = {}
    functions_read = 0
    unreachable: dict[str, tuple[int, int]] = {}
    for path in _rust_source_files():
        bodies = _rust_top_level_function_bodies(path)
        in_file = len(call.findall(_rust_code(path)))
        attributed = sum(len(call.findall(body)) for body in bodies.values())
        if attributed != in_file:
            unreachable[path.relative_to(REPO_ROOT).as_posix()] = (in_file, attributed)
        for name, body in bodies.items():
            functions_read += 1
            if _RUST_SETTINGS_WINDOW_PATTERN.search(body) and call.search(body):
                reaching[name] = path.relative_to(REPO_ROOT).as_posix()

    assert functions_read, (
        f"no top-level fn was read under {RUST_SOURCE_DIR.relative_to(REPO_ROOT).as_posix()}; "
        "the walk has gone blind and the assertion below would pass on nothing"
    )
    assert unreachable == {}, (
        f"this reader names top-level fns only, so a .{verb}() written inside an impl block or "
        "a nested mod would be invisible to the assertion below rather than caught by it; these "
        f"files call .{verb}() more often than it can attribute (in file, attributed): "
        f"{unreachable}"
    )
    assert sorted(reaching) == [helper], (
        f"only the helper that emits '{event}' may {verb} the settings window, or the page "
        "goes on drawing a window state that never happened (ADR 089); it is "
        f"{verb}n from {sorted(reaching.items())}"
    )


def test_no_other_rust_function_shows_the_settings_window() -> None:
    """A fn that names the settings window and shows it must be the announcing helper.

    What this establishes: across every top-level ``fn`` under src-tauri/src/,
    the ones whose body holds both a literal ``get_webview_window("settings")``
    and a ``.show()`` are exactly ``show_settings``. Each file's ``.show()``
    calls are counted twice — in the file and inside the fns this reader can
    name — so a call it cannot attribute fails here rather than passing unseen.

    What it does not establish: nothing here analyses Rust, so a show whose
    window arrived from a helper, from managed state or from a binding made in
    another fn is invisible to the key. It narrows where a settings show can be
    written; it does not prove the announcement complete (ADR 089).
    """
    _assert_only_one_fn_reaches_the_settings_window("show", "show_settings", "settings-shown")


def test_no_other_rust_function_hides_the_settings_window() -> None:
    """A fn that names the settings window and hides it must be the announcing helper.

    The mirror of the walk above, under the same limits: it narrows where a
    settings hide can be written and does not prove the announcement complete.
    """
    _assert_only_one_fn_reaches_the_settings_window("hide", "hide_settings", "settings-hidden")


def test_the_rust_reader_does_not_take_prose_for_code(tmp_path: Path) -> None:
    """A call named in a Rust comment is not a call site; a ``//`` in a string is not a comment.

    Both pins above map ``.show()`` calls to the function holding them, and a
    doc comment sits above its ``fn`` and so inside none of them: prose naming
    the call would be counted in the file and placed in no function, failing a
    correct file. Blanking to spaces rather than to nothing is what keeps the
    reported positions truthful.
    """
    source = tmp_path / "prose.rs"
    source.write_text(
        '/// Calls .show() and emits "settings-shown".\n'
        "fn documented() {\n"
        '    let endpoint = "http://127.0.0.1:9377/a//b";\n'
        "    /* .show() again,\n"
        '       and "settings-shown" again */\n'
        "}\n",
        encoding="utf-8",
    )

    code = _rust_code(source)

    assert ".show(" not in code, (
        "a .show() written in Rust prose is still read as a call site, so a doc "
        f"comment naming the call fails the show pins on a correct file: {code!r}"
    )
    assert "settings-shown" not in code, (
        "an event name written in Rust prose is still read as an emit, so the pin "
        f"passes on a helper that has stopped emitting it: {code!r}"
    )
    assert '"http://127.0.0.1:9377/a//b"' in code, (
        "the // inside a string literal was read as the start of a comment, which "
        f"silently deletes the rest of a line of real code: {code!r}"
    )
    assert len(code.splitlines()) == 6, (
        "blanking changed the line count, so every position this module reports "
        f"about a Rust file is off by however many comment lines precede it: {code!r}"
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


def test_every_capture_incident_token_is_spelled_the_same_in_both_languages() -> None:
    """The widget picks the sentence it shows a user off these tokens.

    The backend names what went wrong with a meeting capture and the widget
    decides what that means to a user, so a token only one side knows degrades
    the marker with no explanation -- or, the other way round, leaves a
    sentence nothing can ever trigger. Neither is visible to `tsc` or to
    `mypy`.

    Mutation-checked: renaming `STORAGE_LOW`'s value in `meeting_recorder.py`
    and leaving `src/contracts.ts` alone fails this test printing both sets.
    """
    python_tokens = set(
        re.findall(
            r'^    [A-Z_]+ = "([a-z_]+)"$',
            _class_body(MEETING_RECORDER_PY, "CaptureIncident"),
            re.MULTILINE,
        )
    )
    declaration = _extract(
        CONTRACTS_TS, r"export const CAPTURE_INCIDENTS = \[([^\]]*)\] as const;"
    )[0]
    typescript_tokens = set(re.findall(r'"([a-z_]+)"', declaration))
    assert python_tokens == typescript_tokens, (
        f"{MEETING_RECORDER_PY.name} can report {sorted(python_tokens)} but "
        f"{CONTRACTS_TS.name} carries {sorted(typescript_tokens)}"
    )
    assert len(python_tokens) >= 5, (
        "the extractor matched fewer tokens than CaptureIncident declares, so "
        "it is no longer reading the enum it exists to pin"
    )


def _class_body(path: Path, name: str) -> str:
    """The indented body of one class, so a scan cannot stray into its siblings.

    `meeting_recorder.py` holds two `str, Enum` classes and a token pin that
    read the whole file collected `MeetingState`'s members as well.
    """
    text = _read(path)
    start = text.index(f"class {name}(")
    body = text[start:]
    end = re.search(r"^(?:class |def )", body[1:], re.MULTILINE)
    return body if end is None else body[: end.start() + 1]


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


def _handler_macro_bodies(path: Path) -> list[str]:
    """The text inside each ``tauri::generate_handler![...]`` in one file.

    Bracket depth is counted rather than matched with a regex. A non-greedy
    bracket group ends at the first closing bracket, which is the one closing
    an attribute written on an entry -- ``#[cfg(desktop)]`` is legal inside the
    macro, ``tauri-macros`` parses it, and the truncated body then yields no
    entry at all, so every registration in the file disappeared and the caller
    reported "no macro was found". Depth counting reads the whole body instead.

    An unterminated macro is a file that does not compile; it raises here
    naming the file rather than contributing a silently empty body.
    """
    text = _read(path)
    bodies: list[str] = []
    start = text.find(_RUST_HANDLER_OPENING)
    while start != -1:
        cursor = start + len(_RUST_HANDLER_OPENING)
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "[":
                depth += 1
            elif text[cursor] == "]":
                depth -= 1
            cursor += 1
        assert not depth, (
            f"{path.relative_to(REPO_ROOT).as_posix()} opens a tauri::generate_handler! "
            "macro that is never closed; the file cannot compile"
        )
        bodies.append(text[start + len(_RUST_HANDLER_OPENING) : cursor - 1])
        start = text.find(_RUST_HANDLER_OPENING, cursor)
    return bodies


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

    An attribute on an entry is stripped before the path is read, so a gated
    registration still contributes its name. What that does **not** distinguish
    is which platform the gate names: a command registered only under
    ``#[cfg(mobile)]`` counts as registered here while being unreachable on a
    desktop build. Pinning that needs the build's own target, which this module
    has no way to read -- it reads text and runs on the Linux CI box with no
    Rust toolchain at all (ADR 045).

    Mutation-checked: gating one entry with ``#[cfg(desktop)]`` keeps this test
    green and every other test in the module green, where before the whole
    registration set came back empty and the failure named no macro at all; and
    a genuinely unregistered command still fails naming that command.
    """
    registered: set[str] = set()
    for path in _rust_source_files():
        for body in _handler_macro_bodies(path):
            for entry in body.split(","):
                bare = re.sub(_RUST_ENTRY_ATTRIBUTE_PATTERN, "", entry).strip()
                match = re.fullmatch(_RUST_HANDLER_ENTRY_PATTERN, bare)
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
    ``cfg!(debug_assertions)`` or that same force-dev variable, and uses it for
    exactly one thing, the sidecar log directory. Rust does not decide Python's
    data root -- it passes the force-dev flag and lets Python resolve. The
    failure this pin prevents is the one backend.rs promises out loud: the
    sidecar log landing in a
    different directory from the sidecar's own history and settings files.

    **Two ways the two sides can still land in different directories at
    runtime, neither visible to any text pin:** that ``cfg!(debug_assertions)``
    and ``sys.frozen`` agree on a given launch is a runtime property; and on a
    Unix host with no ``HOME``, ``posixpath.expanduser`` falls back to
    ``pwd.getpwuid()`` while ``backend.rs``'s ``home_dir`` has no equivalent in
    ``std`` and writes no log at all rather than guessing.

    The other two this docstring used to list were closed by JS-131 --
    ``JUSTSAY_DATA_DIR`` ignored by the shell, and an externally set
    ``JUSTSAY_FORCE_DEV_DATA_DIR`` splitting a release build -- and the test
    below is what keeps them closed.

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


def test_the_shell_reads_every_data_directory_variable_the_backend_reads() -> None:
    """The sidecar log follows ``resolve_app_data_root()``'s order, not its own.

    The two sides resolve the data root independently -- Python in
    ``app_paths.py``, Rust in ``backend.rs`` for the one thing it writes -- and
    the sibling test above compares only the two directory *names*. Names
    agreeing is not enough: before JS-131 both names matched while the shell
    ignored ``JUSTSAY_DATA_DIR`` entirely, so the file someone opens when the
    backend will not start was the one file left behind in the default
    location.

    Three things are pinned, and the third is the one an earlier draft of this
    test missed. Every variable ``app_paths.py`` declares appears in
    ``backend.rs``; the override is read at the site that decides the log
    directory; and **no data-directory variable name appears in ``backend.rs``
    that Python does not declare**. Without that third rule, renaming a
    variable in ``app_paths.py`` reds only the one Rust line this test names,
    and the two ``.env()`` calls that export the old name to the child stay
    green -- which is the same split JS-131 closed, arrived at from the other
    direction.

    The names are read out of ``app_paths.py`` by pattern rather than written
    again here, so a *third* variable added there is covered without editing
    this test -- which is what its own name promises.

    What this cannot see is whether the Rust side *honours* what it reads.
    ``backend.rs``'s ``#[cfg(test)]`` tests cover each branch of the order, and
    they run under ``npm run test:rust``, which no CI job invokes yet (open
    question B2) -- which is why the reads are pinned here, in a job that does.

    The Rust patterns tolerate arbitrary whitespace, following the sibling
    test: ``cargo fmt`` runs in no CI job here, so an indentation change must
    not be reported as a missing binding.

    Mutation-checked three times, each applied alone: deleting the
    ``JUSTSAY_DATA_DIR`` read from ``sidecar_log_dir`` fails this test naming
    that function; reverting ``force_dev_data_dir`` to ``cfg!(debug_assertions)``
    alone fails it naming that binding; and renaming ``_FORCE_DEV_ENV_VAR``'s
    value in ``app_paths.py`` fails it naming the stale literals left in
    ``backend.rs``.
    """
    declared = dict(
        re.findall(r'^(_\w+_ENV_VAR) = "([^"]*)"$', _read(APP_PATHS_PY), re.MULTILINE)
    )
    assert declared, f"no _*_ENV_VAR declarations found in {APP_PATHS_PY.name}"
    rust = _read(BACKEND_RS)
    backend_rel = BACKEND_RS.relative_to(REPO_ROOT).as_posix()

    unread = sorted(name for name in declared.values() if f'"{name}"' not in rust)
    assert not unread, (
        f"{APP_PATHS_PY.name} resolves the data root from {unread}, which {backend_rel} "
        "never mentions, so the sidecar log would be written outside the directory holding "
        "the history and settings it describes"
    )

    strays = sorted(
        set(re.findall(r'"(JUSTSAY_[A-Z_]*DATA_DIR[A-Z_]*)"', rust)) - set(declared.values())
    )
    assert not strays, (
        f"{backend_rel} names {strays}, which {APP_PATHS_PY.name} does not declare -- a "
        "renamed variable left behind here, so the two sides resolve different directories"
    )

    resolver = re.search(r"^fn sidecar_log_dir\(.*?^\}", rust, re.MULTILINE | re.DOTALL)
    assert resolver, f"{backend_rel} no longer defines fn sidecar_log_dir"
    data_dir_var = declared["_DATA_DIR_ENV_VAR"]
    assert data_dir_var in resolver.group(0), (
        f"{APP_PATHS_PY.name} resolves the data root from {data_dir_var!r} first, but "
        f"{backend_rel}'s sidecar_log_dir does not read it"
    )

    force_dev = re.search(r"^\s*let force_dev_data_dir\s*=.*?;$", rust, re.MULTILINE | re.DOTALL)
    assert force_dev, f"{backend_rel} no longer binds force_dev_data_dir"
    force_dev_var = declared["_FORCE_DEV_ENV_VAR"]
    assert force_dev_var in force_dev.group(0), (
        f"{APP_PATHS_PY.name} lets {force_dev_var!r} force the development directory, but "
        f"{backend_rel} picks its name from the build profile alone, so a release build that "
        "inherits that variable writes the log under the production name while the backend "
        "resolves the development one"
    )


def _swift_code() -> str:
    """``main.swift`` with its comment bodies blanked out.

    ``macos/JustSayAudioTap/**`` is the one tree ``CLAUDE.md`` exempts from
    the comment ban and encourages comments in: the Swift cannot be compiled
    anywhere in this repository, so the constraints it obeys are written down
    beside it. A scan that reads raw lines therefore fails on prose. One more
    ``///`` sentence naming ``flushWholeBlocks()`` in ``stop()``'s existing
    doc block would redden this suite with the Swift behaviour untouched, and
    whoever wrote that sentence has no way to see the failure is spurious.

    Both spellings are blanked. ``/* ... */`` is not a variant this can skip:
    a block comment holding a line that is a closing brace at a function's own
    indent -- prose about the code it sits in, or code commented out -- ends
    that function as far as the reader below is concerned, which cut
    ``flushWholeBlocks`` from 521 characters to 21 and turned
    ``_swift_call_sites("writeAll")`` into ``['writeHeader', '<top level>']``.
    A block comment merely naming ``writeAll(`` counted as a call site by the
    same omission.

    Comment bodies become spaces rather than disappearing, so every line and
    every column stays where it was and the indentation the reader below
    captures is unchanged. Newlines inside a block comment are kept for the
    same reason. String literals are matched first and passed through, so a
    ``//`` inside a quoted value is not read as a comment.
    """

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("/"):
            return _NOT_A_NEWLINE.sub(" ", token)
        return token

    return _SWIFT_COMMENT_OR_STRING_PATTERN.sub(blank, _read(AUDIO_TAP_SWIFT))


def _swift_function_body(name: str) -> str:
    r"""One Swift function of ``main.swift``, from its signature to its close.

    Bounded by the closing brace at the signature's own indentation rather
    than by a brace count, because nothing here parses Swift and a brace
    counter over string literals and generics would be a parser pretending
    not to be one. Every brace nested inside a function in this file closes
    further in, so the first line that is exactly the signature's indent
    followed by ``}`` is that function's own end.

    It used to end at the *next* ``func`` at that indent, which is not the
    same thing: the last function of a type has no successor, so its body ran
    to the end of the file and any assertion over it degraded into a
    whole-file grep. ``stop()`` is that last function, and the
    ``queue.sync { flushWholeBlocks() }`` pin below passed on a file where
    that flush had been moved out of ``stop()`` into the top-level code under
    the class -- which is the SIGTERM flush gone. Measured on the tree that
    fixed it: 1116 characters before, 601 after. It was the second time the
    same degradation shipped: ``flushWholeBlocks`` is followed by a ``///``
    doc comment rather than a blank line, which once made the old terminator
    overrun it too (1855 characters before, 726 after). Closing on the brace
    ends both, because a function's own close is in the file whatever comes
    after it.

    The indent is captured with ``[ \t]`` and not ``\s``: under
    ``re.MULTILINE`` the latter matches the newline of the blank line *before*
    the signature, so the captured indent began with one and the terminator
    then demanded a blank line immediately before it.
    """
    source = _swift_code()
    opening = re.search(
        rf"^([ \t]*)(?:private\s+)?func {re.escape(name)}\b", source, re.MULTILINE
    )
    assert opening, f"{AUDIO_TAP_SWIFT.name} no longer defines func {name}"
    rest = source[opening.end() :]
    closing = re.search(rf"\n{opening.group(1)}\}}", rest)
    assert closing, (
        f"{AUDIO_TAP_SWIFT.name}'s func {name} has no closing brace at its own "
        f"indent, so this reader cannot say where it ends"
    )
    return rest[: closing.start()]


def _swift_call_sites(callee: str) -> list[str]:
    """Every call to ``callee`` in ``main.swift``, named by what encloses it.

    Sorted function names, with ``"<top level>"`` standing for a call outside
    every function -- ``main.swift`` runs its last twenty lines at file scope,
    so "which function calls this" is not on its own the whole answer. The
    callee's own definition is never counted: ``_swift_function_body`` starts
    a body after the name in its signature, so the only way ``callee(`` shows
    up inside it is a recursive call.
    """
    call = re.compile(rf"(?<!\w){re.escape(callee)}\(")
    code = _swift_code()
    enclosed: list[str] = []
    for name in _SWIFT_FUNCTION_PATTERN.findall(code):
        enclosed.extend([name] * len(call.findall(_swift_function_body(name))))
    everywhere = len(call.findall(code)) - len(
        re.findall(rf"func {re.escape(callee)}\(", code)
    )
    return sorted(enclosed) + ["<top level>"] * (everywhere - len(enclosed))


def _python_function_body(path: Path, name: str) -> str:
    r"""One Python function, from its ``def`` to the next line at that indent.

    ``[ \t]`` rather than ``\s`` for the same reason as the Swift reader
    above, where the difference was load-bearing: a captured indent starting
    with a newline makes the terminator demand a blank line.

    The terminator matches an explicit newline rather than a ``^`` under
    ``re.MULTILINE``, because ``^`` also matches the start of the string --
    which for a module-level function, whose indent is empty, is the middle of
    its own signature line, so the body came back empty.
    """
    source = _read(path)
    opening = re.search(rf"^([ \t]*)def {re.escape(name)}\b", source, re.MULTILINE)
    assert opening, f"{path.name} no longer defines def {name}"
    rest = source[opening.end() :]
    following = re.search(rf"\n{opening.group(1)}\S", rest)
    return rest[: following.start()] if following else rest


def test_the_macos_tap_helper_and_its_reader_agree_on_the_header() -> None:
    """The helper writes this line; ``parse_tap_header`` reads it. Four parties.

    ``macos/JustSayAudioTap`` is Swift, is built only by the macOS release job
    and cannot be compiled -- let alone run -- anywhere in this repository.
    ``CLAUDE.md`` exempts it from the comment ban for exactly that reason and
    says what the exemption costs: its file header and the module docstring in
    ``app/audio/macos_tap.py`` "are the two halves of one contract" and
    "nothing in the repository checks the two against each other". This is that
    check, and it is why the two prose halves are compared here as well as the
    code that implements them -- a header key renamed in the Swift and in its
    own comment, with the Python left alone, is the drift that ships.

    The tap helper is the only source of macOS system audio, so a header the
    Python refuses is a meeting recorded with the microphone alone, reported
    through `SystemAudioUnavailableError` at start time at best.
    """
    emitted = set(re.findall(_SWIFT_HEADER_KEY_PATTERN, _swift_function_body("writeHeader")))
    required = set(
        re.findall(
            _PYTHON_HEADER_KEY_PATTERN, _python_function_body(MACOS_TAP_PY, "parse_tap_header")
        )
    )

    assert emitted == required, (
        f"{AUDIO_TAP_SWIFT.name} writes the header keys {sorted(emitted)} and "
        f"parse_tap_header reads {sorted(required)}; the helper is the only "
        f"source of macOS system audio and this line is the only thing that "
        f"describes its bytes"
    )

    declared_format = re.search(
        _SWIFT_HEADER_FORMAT_PATTERN, _swift_function_body("writeHeader")
    )
    assert declared_format, f"{AUDIO_TAP_SWIFT.name} no longer writes a format literal"
    accepted_format = re.search(r'^SAMPLE_FORMAT = "([^"]+)"$', _read(MACOS_TAP_PY), re.MULTILINE)
    assert accepted_format, "macos_tap.py no longer declares SAMPLE_FORMAT"
    assert declared_format.group(1) == accepted_format.group(1), (
        f"the helper declares format {declared_format.group(1)!r} and the reader "
        f"accepts only {accepted_format.group(1)!r}, so every capture is refused "
        f"at startup"
    )

    documented = []
    for path in (AUDIO_TAP_SWIFT, MACOS_TAP_PY):
        written_down = re.search(_DOCUMENTED_HEADER_PATTERN, _read(path))
        assert written_down, (
            f"{path.name} no longer writes down the stdout line, which is the "
            f"half of this contract a reader of the other file goes by"
        )
        documented.append(json.loads(written_down.group(1)))
    assert documented[0] == documented[1], (
        f"the helper's header comment and the macos_tap.py docstring describe "
        f"different stdout lines: {documented[0]} against {documented[1]}"
    )
    assert set(documented[0]) == emitted, (
        f"the two prose halves of the contract describe keys {sorted(documented[0])} "
        f"while writeHeader emits {sorted(emitted)}"
    )
    assert documented[0].get("format") == accepted_format.group(1), (
        f"the documented header declares format {documented[0].get('format')!r} "
        f"and macos_tap.py accepts {accepted_format.group(1)!r}"
    )


def test_the_macos_tap_helper_and_its_reader_frame_blocks_the_same_way() -> None:
    """Both sides count interleaved frames off the channel count in the header.

    The helper writes ``blockFrames * channels`` samples at a time and the
    reader asks the pipe for exactly that many bytes. Nothing in the byte
    stream marks a block boundary, so the two arithmetics *are* the framing: a
    Swift edit that multiplied by something else, or sent a wider sample, would
    hand the reader blocks it either splits in the wrong place -- recording
    audio that is silently wrong rather than absent -- or refuses as malformed.
    Neither is observable from this side of the pipe without this test.

    The channel count both sides multiply by is the one the header declares,
    which is what makes the two agree by construction rather than by luck. The
    helper refuses to capture a buffer disagreeing with it, twice over, and
    those two guards are pinned below for the same reason the arithmetic is.

    So is the way the bytes leave the helper, which is what makes the reader's
    framing safe to trust rather than merely checked. ``flushWholeBlocks``
    hands ``writeAll`` a whole number of blocks, and ``writeAll`` loops until
    every byte of them is out; the only thing it does instead is stop writing
    altogether. A helper that wrote a partial frame and carried on would slip
    the stream by an offset the reader re-cuts every later block at, which
    nothing in the bytes reveals and no runtime guard on this side can see --
    `MalformedCaptureBlockError` exists for the shape, and could not have
    caught it. These three literals are where that is checked instead.
    """
    swift_source = _read(AUDIO_TAP_SWIFT)
    flush = _swift_function_body("flushWholeBlocks")
    assert _SWIFT_BLOCK_SAMPLES in flush, (
        f"{AUDIO_TAP_SWIFT.name} no longer sizes a block as "
        f"{_SWIFT_BLOCK_SAMPLES!r}, so the helper and macos_tap.py cut the "
        f"stream in different places and nothing in the bytes says so"
    )

    for literal in _SWIFT_WHOLE_BLOCK_WRITES:
        assert literal in swift_source, (
            f"{AUDIO_TAP_SWIFT.name} no longer writes {literal!r}, so the helper "
            f"can leave a partial frame in the stream and carry on -- every block "
            f"the reader cuts after it is misframed, and nothing in the bytes or "
            f"on the Python side says so"
        )

    deliver = _python_function_body(MACOS_TAP_PY, "_deliver_until_refused")
    assert _PYTHON_BLOCK_BYTES in deliver, (
        f"macos_tap.py no longer reads {_PYTHON_BLOCK_BYTES!r} bytes per block"
    )
    assert _PYTHON_DEINTERLEAVE in deliver, (
        f"macos_tap.py no longer deinterleaves with {_PYTHON_DEINTERLEAVE!r}, so "
        f"the channel count it read blocks by and the one it splits frames by "
        f"can differ"
    )

    accepted_format = re.search(r'^SAMPLE_FORMAT = "([^"]+)"$', _read(MACOS_TAP_PY), re.MULTILINE)
    assert accepted_format, "macos_tap.py no longer declares SAMPLE_FORMAT"
    spelling = _TAP_SAMPLE_SPELLINGS.get(accepted_format.group(1))
    assert spelling, (
        f"the contract declares format {accepted_format.group(1)!r}, which this "
        f"test knows no Swift spelling for, so nothing below compares anything"
    )
    source_constants = _read(SYSTEM_SOURCE_PY)
    dtype = re.search(r'^SAMPLE_DTYPE = "([^"]+)"$', source_constants, re.MULTILINE)
    sample_bytes = re.search(r"^SAMPLE_BYTES = (\d+)$", source_constants, re.MULTILINE)
    assert dtype and sample_bytes, (
        "system_source.py no longer declares SAMPLE_DTYPE and SAMPLE_BYTES"
    )
    read_as = (dtype.group(1), int(sample_bytes.group(1)))
    declared = _TAP_PYTHON_SPELLINGS.get(accepted_format.group(1))
    assert read_as == declared, (
        f"the contract declares format {accepted_format.group(1)!r}, which both "
        f"sources must read as {declared}, but system_source.py declares "
        f"{read_as} -- the block arithmetic above is pinned through those two "
        f"names and would be pinned to nothing"
    )

    element, width = spelling
    assert element in swift_source and width in swift_source, (
        f"the contract declares {accepted_format.group(1)!r}, which is 4 bytes a "
        f"sample, but {AUDIO_TAP_SWIFT.name} no longer buffers {element!r} or "
        f"measures {width!r} -- every block the reader asks for would be the "
        f"wrong length"
    )

    for function, guard in _SWIFT_CHANNEL_GUARDS:
        assert guard in _swift_function_body(function), (
            f"{AUDIO_TAP_SWIFT.name}'s {function} no longer checks {guard!r}, so "
            f"a buffer whose channel count disagrees with the header would be "
            f"written into the stream and the reader has no way to notice"
        )


def test_the_macos_tap_helper_writes_its_stdout_from_one_writer_at_a_time() -> None:
    """Every write to the helper's stdout is serialised, so none interleave.

    `_ran_out` in macos_tap.py decides a capture was cut short by measuring a
    partial block against a whole one, and the test above pins the three
    literals that make every write a whole number of blocks. Those literals
    are only half of what that check rests on. The other half is that one
    write finishes before the next begins, and nothing read it until here.

    `writeAll` is the write -- it is the only call to `write(STDOUT_FILENO,
    ...)` in the helper -- and it has exactly two call sites. One is
    `flushWholeBlocks`, reached from `consume` on the IOProc block, which
    `AudioDeviceCreateIOProcIDWithBlock` is told to dispatch on the tap's own
    queue rather than on a real-time thread of its own, and reached again
    from `stop()` -- the SIGTERM teardown, which arrives on the main queue --
    through `queue.sync`. That queue is created with a label and nothing
    else, so it is serial, and those two take turns. The other is
    `writeHeader`, which `start()` calls once, before `startIOProc()` has
    created anything that could be writing beside it.

    Make the queue `.concurrent`, hand the IOProc `nil` and let Core Audio
    pick the thread, or add a third `writeAll` from anywhere -- a status
    trailer on teardown, a debug dump -- and two writes can land inside each
    other. A block is 8192 bytes against a `PIPE_BUF` that Darwin's
    `sys/syslimits.h` puts at 512, so `writeAll` finishes neither of them
    atomically and the reader is handed the two spliced into one full-sized
    block. Every later block is then cut in the wrong place, the short-read
    check never fires because nothing ever reads short, and the recording is
    silently wrong rather than absent -- the failure the reader cannot see
    from its side of the pipe, which is why it is checked from this one.
    """
    assert _SWIFT_TAP_SERIAL_QUEUE.search(_swift_code()), (
        f"{AUDIO_TAP_SWIFT.name} no longer declares its io queue as a plain "
        f"labelled DispatchQueue -- a concurrent one lets the teardown flush "
        f"interleave with the IOProc's, and the reader is handed two "
        f"half-blocks spliced into one"
    )

    assert _SWIFT_IOPROC_ON_THE_TAP_QUEUE.search(_swift_function_body("startIOProc")), (
        f"{AUDIO_TAP_SWIFT.name} no longer hands the IOProc the tap's own "
        f"queue, so its blocks run on a thread the teardown flush does not "
        f"take turns with"
    )

    assert _swift_call_sites("writeAll") == _SWIFT_STDOUT_WRITERS, (
        f"{AUDIO_TAP_SWIFT.name} writes stdout from {_swift_call_sites('writeAll')} "
        f"and not from {_SWIFT_STDOUT_WRITERS} -- every write has to be one of "
        f"the two this test can account for, and a third is a writer nothing "
        f"serialises against the other two"
    )

    assert _swift_call_sites("flushWholeBlocks") == _SWIFT_FLUSH_CALLERS, (
        f"{AUDIO_TAP_SWIFT.name} flushes from "
        f"{_swift_call_sites('flushWholeBlocks')} and not from "
        f"{_SWIFT_FLUSH_CALLERS} -- the capture-time writer is serialised only "
        f"while the IOProc block and stop() are the only two that reach it"
    )

    assert _SWIFT_FLUSH_FROM_STOP.search(_swift_function_body("stop")), (
        f"{AUDIO_TAP_SWIFT.name}'s stop no longer flushes through queue.sync, "
        f"so the SIGTERM flush writes from the main queue while the IOProc is "
        f"writing from its own"
    )

    assert _swift_call_sites("writeHeader") == ["start"], (
        f"{AUDIO_TAP_SWIFT.name} writes its header from "
        f"{_swift_call_sites('writeHeader')} -- the header is safe to write off "
        f"the tap queue only because start() is the one caller and nothing is "
        f"capturing yet when it runs"
    )

    start = _swift_function_body("start")
    assert start.index("writeHeader(") < start.index("startIOProc("), (
        f"{AUDIO_TAP_SWIFT.name}'s start() creates the IOProc before it writes "
        f"the header, so the first blocks can reach stdout while the header "
        f"write is still going"
    )


def test_the_macos_tap_helper_and_its_reader_agree_on_the_command_line() -> None:
    """One flag, spelled by the spawner and parsed by the helper.

    A helper that fell back to its own default block size would produce blocks
    the reader is not asking for, which is the framing failure above arrived at
    through the argument vector instead of the header.
    """
    spawn = _python_function_body(MACOS_TAP_PY, "_spawn")
    parse = _swift_function_body("parseBlockFrames")

    assert f'"{_TAP_BLOCK_FRAMES_FLAG}"' in spawn, (
        f"macos_tap.py no longer passes {_TAP_BLOCK_FRAMES_FLAG!r} to the helper"
    )
    assert f'"{_TAP_BLOCK_FRAMES_FLAG}"' in parse, (
        f"{AUDIO_TAP_SWIFT.name} no longer parses {_TAP_BLOCK_FRAMES_FLAG!r}, so "
        f"it falls back to its own default block size and the reader waits for "
        f"bytes that arrive in different-sized pieces"
    )

    declared = []
    for path in (AUDIO_TAP_SWIFT, MACOS_TAP_PY):
        written_down = re.search(_DOCUMENTED_ARGV_PATTERN, _read(path))
        assert written_down, (
            f"{path.name} no longer writes down the command line, the half of this "
            f"contract a reader of the other file goes by"
        )
        declared.append(written_down.group(0).strip())
    assert declared[0] == declared[1], (
        f"the helper's header and the macos_tap.py docstring declare different "
        f"command lines: {declared[0]!r} against {declared[1]!r}"
    )
