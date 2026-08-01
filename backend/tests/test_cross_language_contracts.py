"""Four values exist in two or three languages at once; the copies must agree.

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

ADR 045 records why these values are pinned rather than generated.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PY = REPO_ROOT / "backend" / "app" / "core" / "config.py"
CONSTANTS_PY = REPO_ROOT / "backend" / "app" / "core" / "constants.py"
AUDIO_FORMATS_PY = REPO_ROOT / "backend" / "app" / "core" / "audio_formats.py"
CONTRACTS_TS = REPO_ROOT / "src" / "contracts.ts"
LIB_RS = REPO_ROOT / "src-tauri" / "src" / "lib.rs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract(path: Path, pattern: str) -> list[str]:
    matches = re.findall(pattern, _read(path), re.MULTILINE)
    assert matches, f"no declaration matching {pattern!r} found in {path}"
    values: list[str] = []
    for match in matches:
        values.extend([match] if isinstance(match, str) else match)
    return values


_PORT_SITES: dict[Path, str] = {
    CONFIG_PY: r"^    port: int = (\d+)$",
    CONTRACTS_TS: r"^export const BACKEND_PORT = (\d+);$",
    REPO_ROOT / "src-tauri" / "src" / "backend.rs": r"^pub const PORT: u16 = (\d+);$",
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
    ("src-tauri/src", "*.rs"),
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


def _scanned_files() -> list[Path]:
    found: list[Path] = []
    for directory, glob in _SCAN_GLOBS:
        found.extend(sorted((REPO_ROOT / directory).glob(glob)))
    assert found, "the enumerated scan set matched no files at all"
    return found


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

    strays: list[str] = []
    quoted = re.compile(f"[\"']{re.escape(python_value)}[\"']")
    for path in _scanned_files():
        if path in (CONSTANTS_PY, CONTRACTS_TS) or path.name.endswith((".test.ts", "_test.py")):
            continue
        if path.name.startswith("test_"):
            continue
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            if quoted.search(line):
                strays.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")
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


def _typescript_event_sites() -> list[tuple[Path, str, int, str]]:
    sites: list[tuple[Path, str, int, str]] = []
    for path in sorted((REPO_ROOT / "src").glob("**/*.ts")):
        if path.name.endswith(".test.ts") or path == CONTRACTS_TS:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            sites.append((path, rel, lineno, line))
    return sites


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

    typescript_lines = _typescript_event_sites()

    used: list[tuple[str, int, str]] = []
    for _, rel, lineno, line in typescript_lines:
        for pattern in _TYPESCRIPT_EVENT_PATTERNS:
            for name in re.findall(pattern, line):
                used.append((rel, lineno, name))
    rust_emits: list[tuple[str, int, str]] = []
    for path in sorted((REPO_ROOT / "src-tauri" / "src").glob("*.rs")):
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
            for _, rel, lineno, line in typescript_lines
            if any(
                re.search(template.format(constant=constant), line)
                for template in _TYPESCRIPT_EMITTER_TEMPLATES
            )
        ]
        emitters += [f"{rel}:{lineno}" for rel, lineno, name in rust_emits if name == value]
        listeners = [
            f"{rel}:{lineno}"
            for _, rel, lineno, line in typescript_lines
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
