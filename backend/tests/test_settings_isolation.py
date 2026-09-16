"""The conftest guarantee that `sync_to_runtime`'s writes do not outlive a test.

`app.core.config.settings` is a process-wide singleton, and
`user_settings.sync_to_runtime()` writes onto it directly, so `monkeypatch`
cannot intercept it and a test driving it must restore the values itself.
Until 2026-08-25 `_reset_settings` restored two of the twelve fields it writes,
so ten -- including `initial_prompt`, `whisper_model_size` and both cloud API
keys -- leaked into whatever test ran next.

Two things are pinned here. That the restore round-trips, and that its field
list still matches `sync_to_runtime`'s own source: the second is what stops the
list going stale the next time a field is added there.

Since spec 165 the singleton is *defined* in `app/config.py` and re-exported by
`app/core/config.py`, so two spellings reach one object. `app.core.config` is
the spelling used throughout this file and everywhere else in the repository;
`app.config` appears below only in the two tests that exist to pin the
relationship between the two modules. ADR 076 records why both exist.
"""

import ast
from pathlib import Path

import app.config as composition_root
import app.core.config as core_config_module
import app.preferences.user_settings as user_settings_module
from app.core.config import settings
from tests.conftest import (
    RUNTIME_SETTINGS_FIELDS_WRITTEN_BY_SYNC,
    restore_runtime_settings,
    snapshot_runtime_settings,
)

_SENTINELS: dict[tuple[str, str], object] = {
    ("stt", "mode"): "local",
    ("stt", "whisper_model_size"): "tiny",
    ("stt", "whisper_device"): "cpu",
    ("stt", "cloud_routing_threshold"): 999.0,
    ("stt", "engine"): "whisper_cpp",
    ("stt", "initial_prompt"): "sentinel glossary",
    ("stt", "gemini_api_key"): "sentinel-gemini-key",
    ("stt", "groq_api_key"): "sentinel-groq-key",
    ("embeddings", "ollama_host"): "http://sentinel:1",
}


def _fields_assigned_by_sync_to_runtime() -> dict[str, set[str]]:
    """Every `settings.<child>.<field> = ...` target inside `sync_to_runtime`.

    Read from the source with `ast` rather than by calling the function: the
    point is to catch a field someone adds there later, which a behavioural
    call could only notice if the test already knew to look for it.
    """
    source = Path(user_settings_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sync_to_runtime"
    )

    assigned: dict[str, set[str]] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "settings"
            ):
                assigned.setdefault(target.value.attr, set()).add(target.attr)
    return assigned


def test_restore_list_covers_every_field_sync_to_runtime_writes():
    assigned = _fields_assigned_by_sync_to_runtime()
    restored = {
        child: set(fields)
        for child, fields in RUNTIME_SETTINGS_FIELDS_WRITTEN_BY_SYNC.items()
    }
    written_not_restored = {
        child: sorted(fields - restored.get(child, set()))
        for child, fields in assigned.items()
        if fields - restored.get(child, set())
    }
    restored_not_written = {
        child: sorted(fields - assigned.get(child, set()))
        for child, fields in restored.items()
        if fields - assigned.get(child, set())
    }
    assert assigned == restored, (
        "sync_to_runtime and the conftest restore list disagree. "
        f"Written but not restored: {written_not_restored}. "
        f"Restored but not written: {restored_not_written}"
    )


def test_restore_returns_every_field_to_its_snapshot_value():
    original = snapshot_runtime_settings()
    assert set(original) == set(_SENTINELS), (
        "the sentinel table and the snapshot disagree on which fields exist"
    )

    for (child, field), sentinel in _SENTINELS.items():
        setattr(getattr(settings, child), field, sentinel)

    mutated = snapshot_runtime_settings()
    drifted = {key for key, value in mutated.items() if value == original[key]}
    assert not drifted, (
        f"these sentinels equal the pre-existing value, so they prove nothing: {sorted(drifted)}"
    )

    restore_runtime_settings(original)

    still_wrong = {
        key: (value, original[key])
        for key, value in snapshot_runtime_settings().items()
        if value != original[key]
    }
    assert not still_wrong, f"fields not restored (field: got, expected): {still_wrong}"


def test_both_spellings_of_the_settings_module_bind_one_object():
    """`app.core.config` re-exports what `app.config` defines, so the singleton
    and its class have one identity each rather than two constructions.

    A second `AppSettings()` would read the environment again and diverge the
    moment anything wrote onto either copy — and every `monkeypatch` in this
    suite writes *through* the object (`setattr(settings.audio, ...)`) rather
    than rebinding a module attribute, which only works while there is one
    object to write through.
    """
    assert core_config_module.settings is composition_root.settings
    assert core_config_module.AppSettings is composition_root.AppSettings
    assert settings is composition_root.settings


def test_the_core_config_module_holds_nothing_but_its_re_export():
    """`app/core/config.py` holds a docstring, one import from the composition
    root, and `__all__`. Nothing else, in any position.

    Read from the AST rather than trusted to the docstring saying so. The
    failure this pins is the re-export re-growing into a second composition
    root: a settings class defined here imports a feature package from inside
    `core`, which is what `app/core/config.py` did before spec 165 and what put
    three package cycles through this one module.

    Each arm is exact for a reason it was not when this shipped. Every
    `ImportFrom` was skipped unconditionally, so `from app.audio.config import
    AudioSettings` — the precise regression named above — passed the test meant
    to catch it; only an import from the composition root is accepted now, and
    only one. The docstring is taken from `body[0]` rather than by skipping any
    string constant anywhere, so a bare string cannot ride along after
    `__all__`. Counting the re-exports was not enough on its own: zero of them
    passed as readily as one, so deleting the import left a module whose
    `__all__` named nothing and whose every caller broke, and a star import
    satisfied the same branch without naming what it re-exported. Both are
    rejected explicitly.

    Mutation-checked against the full suite, with the count each one actually
    reddens rather than the count predicted for it. A second `from app.config
    import AppSettings`, a bare string statement appended after `__all__`, and a
    `class ExtraSettings(AppSettings)` named in `__all__`: **one** test each,
    this one. The `app.audio.config` import: **three** — this test, plus
    `test_import_layers`'s feature-package rule and its two-node cycle list,
    because a feature import inside `core/config.py` is all three defects at
    once, which is precisely why it was worth catching here too. Leaving the
    class out of `__all__` fails `test_dead_code_gate` as well, which is the
    mutation being unreferenced rather than anything this assertion claims.
    """
    source = Path(core_config_module.__file__).read_text(encoding="utf-8")
    body = list(ast.parse(source).body)
    if body and ast.get_docstring(ast.Module(body=body, type_ignores=[])):
        body = body[1:]

    offenders = []
    re_exports = 0
    for node in body:
        if (
            isinstance(node, ast.ImportFrom)
            and not node.level
            and node.module == composition_root.__name__
        ):
            if any(alias.name == "*" for alias in node.names):
                offenders.append(
                    f"line {node.lineno}: a star import from {composition_root.__name__}"
                )
                continue
            re_exports += 1
            if re_exports > 1:
                offenders.append(
                    f"line {node.lineno}: a second import from {composition_root.__name__}"
                )
            continue
        if isinstance(node, ast.Assign) and [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ] == ["__all__"]:
            continue
        described = (
            f"an import from {'.' * node.level}{node.module or ''}"
            if isinstance(node, ast.ImportFrom)
            else type(node).__name__
        )
        offenders.append(f"line {node.lineno}: {described}")

    assert re_exports == 1, (
        f"app/core/config.py re-exports the composition root {re_exports} times, "
        "not once. The module exists to be that one doorway; without the import "
        "it is an empty file whose __all__ names nothing, and every call site "
        "spelling app.core.config.settings breaks."
    )

    assert not offenders, (
        "app/core/config.py must stay a docstring, one import from "
        f"{composition_root.__name__} and `__all__`, but holds {offenders}. "
        "Whatever this is belongs in app/config.py, the composition root above "
        "— see ADR 076."
    )
