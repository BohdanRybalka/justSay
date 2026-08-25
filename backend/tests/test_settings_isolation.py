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
"""

import ast
from pathlib import Path

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
    ("llm", "mode"): "local",
    ("llm", "ollama_model"): "sentinel-model",
    ("llm", "ollama_host"): "http://sentinel:1",
    ("llm", "groq_api_key"): "sentinel-groq-key",
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
    assert assigned == restored, (
        "sync_to_runtime writes fields the conftest restore list does not cover "
        "(or vice versa). Writes but not restored: "
        f"{ {c: sorted(f - restored.get(c, set())) for c, f in assigned.items() if f - restored.get(c, set())} }. "
        "Restored but not written: "
        f"{ {c: sorted(f - assigned.get(c, set())) for c, f in restored.items() if f - assigned.get(c, set())} }"
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
