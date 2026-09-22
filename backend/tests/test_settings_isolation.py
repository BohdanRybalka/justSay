"""The conftest guarantee that `sync_to_runtime`'s writes do not outlive a test.

Each feature package owns one process-wide settings instance, and
`user_settings.sync_to_runtime()` writes onto those objects directly, so
`monkeypatch` cannot intercept it and a test driving it must restore the values
itself. Until 2026-08-25 `_reset_settings` restored two of the twelve fields it
writes, so ten -- including `initial_prompt`, `whisper_model_size` and both cloud
API keys -- leaked into whatever test ran next.

Three things are pinned here. That the restore round-trips; that its field list
still matches `sync_to_runtime`'s own source; and that `settings.stt`,
`settings.audio` and `settings.embeddings` are the very objects the packages
read, which is what both the restore and every `monkeypatch` writing through
`settings.audio` rest on. ADR 091 records why the slices live where they do.
"""

import ast
from pathlib import Path

import app.preferences.user_settings as user_settings_module
from app.audio.config import audio_settings
from app.config import settings
from app.embeddings.config import embedding_settings
from app.stt.config import stt_settings
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

_SLICE_VARIABLE_HOLDS = {
    "stt_settings": "stt",
    "audio_settings": "audio",
    "embedding_settings": "embeddings",
}


def _fields_assigned_by_sync_to_runtime() -> dict[str, set[str]]:
    """Every settings field `sync_to_runtime` assigns, keyed by slice name.

    Both spellings that reach a slice are read -- `stt_settings.field` and
    `settings.stt.field` -- so moving a write from one to the other cannot drop
    it from this walk. Read from the source with `ast` rather than by calling
    the function: the point is to catch a field someone adds there later.
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
            if not isinstance(target, ast.Attribute):
                continue
            holder = target.value
            if (
                isinstance(holder, ast.Name)
                and holder.id in _SLICE_VARIABLE_HOLDS
            ):
                assigned.setdefault(_SLICE_VARIABLE_HOLDS[holder.id], set()).add(target.attr)
            elif (
                isinstance(holder, ast.Attribute)
                and isinstance(holder.value, ast.Name)
                and holder.value.id == "settings"
            ):
                assigned.setdefault(holder.attr, set()).add(target.attr)
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


def test_each_composition_root_slice_is_the_instance_its_own_package_holds():
    """`settings.stt`, `settings.audio` and `settings.embeddings` are the
    objects `app.stt`, `app.audio` and `app.embeddings` define, not copies.

    A second instance would read the environment again and diverge the moment
    anything wrote onto either side -- `PUT /stt/mode` writing `local` onto one
    while the pipeline reads `cloud` off the other is exactly the Cloud/Local
    leak this identity rules out (ADR 091).
    """
    assert settings.stt is stt_settings
    assert settings.audio is audio_settings
    assert settings.embeddings is embedding_settings
