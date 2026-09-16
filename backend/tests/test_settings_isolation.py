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
    """`app/core/config.py` declares no class and assigns nothing but `__all__`.

    Read from the AST rather than trusted to the docstring saying so. The
    failure this pins is the re-export re-growing into a second composition
    root: a settings class defined here imports a feature package from inside
    `core`, which is what put three of the four entries into
    `test_import_layers._KNOWN_PACKAGE_CYCLES` before spec 165.

    Mutation-checked: a `class ExtraSettings(AppSettings)` appended to that file
    and named in `__all__` fails this test and no other. Left out of `__all__`
    it fails `test_dead_code_gate` as well, which is the mutation being
    unreferenced rather than anything this assertion is claiming.
    """
    source = Path(core_config_module.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom):
            continue
        if isinstance(node, ast.Assign) and [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ] == ["__all__"]:
            continue
        offenders.append(f"line {node.lineno}: {type(node).__name__}")

    assert not offenders, (
        "app/core/config.py must stay a docstring, an import and `__all__`, "
        f"but holds {offenders}. Whatever this is belongs in app/config.py, the "
        "composition root above — see ADR 076."
    )
