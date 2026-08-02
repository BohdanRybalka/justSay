"""User settings — output_dir validation and runtime sync."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.core.config import AppSettings
from app.preferences import user_settings
from app.transcripts import history


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """See docs/adr/014-lazy-app-data-path-resolution.md: `SETTINGS_DIR` /
    `SETTINGS_PATH` no longer exist as module-level constants to monkeypatch
    -- `JUSTSAY_DATA_DIR` (already set by conftest.py's autouse
    `_isolated_app_data` fixture) is the one supported redirect seam.
    Repointing it here to a nested settings_dir keeps this file's isolation
    distinct from the outer conftest tmp_path, matching this fixture's
    pre-existing shape."""
    settings_dir = tmp_path / "home" / ".justsay"
    settings_dir.mkdir(parents=True)

    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(settings_dir))
    monkeypatch.setattr(user_settings, "_settings", None)
    monkeypatch.setattr(history, "_output_dir", settings_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    yield {"settings_dir": settings_dir}

    with history._lock:
        history._close_conn_locked()



def test_validate_output_dir_rejects_relative():
    with pytest.raises(ValueError, match="absolute"):
        user_settings._validate_output_dir("relative/dir")


def test_validate_output_dir_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        user_settings._validate_output_dir("   ")


def test_validate_output_dir_rejects_non_string():
    with pytest.raises(ValueError, match="non-empty"):
        user_settings._validate_output_dir(42)


def test_validate_output_dir_rejects_when_path_is_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        user_settings._validate_output_dir(str(f))


def test_validate_output_dir_rejects_nonexistent_parent(tmp_path):
    target = tmp_path / "no" / "such" / "parent" / "dir"
    with pytest.raises(ValueError, match="parent directory"):
        user_settings._validate_output_dir(str(target))


def test_validate_output_dir_rejects_when_not_writable(tmp_path, monkeypatch):
    target = tmp_path / "ro"
    target.mkdir()

    def deny_write(self, data):
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "write_bytes", deny_write)
    with pytest.raises(ValueError, match="not writable"):
        user_settings._validate_output_dir(str(target))


def test_validate_output_dir_accepts_valid_dir(tmp_path):
    out = user_settings._validate_output_dir(str(tmp_path))
    assert out == tmp_path.resolve()


def test_validate_output_dir_rejects_forbidden_parent():
    forbidden = "C:/Windows/System32/justsay" if sys.platform == "win32" else "/etc/justsay"
    with pytest.raises(ValueError, match="system directory"):
        user_settings._validate_output_dir(forbidden)



def test_update_output_dir_triggers_history_relocate(tmp_path, monkeypatch):
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    calls: list[Path] = []

    def fake_relocate(target: Path):
        calls.append(target)
        return history.RelocateOutcome.MOVED, None

    monkeypatch.setattr(history, "relocate", fake_relocate)

    user_settings.update_user_settings({"output_dir": str(new_dir)})
    assert len(calls) == 1
    assert calls[0] == new_dir.resolve()



def test_sync_to_runtime_clears_stt_cache_only_on_change(monkeypatch):
    from app.core.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.engine = "auto"
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.mode = ProviderMode.CLOUD
    runtime_settings.llm.ollama_host = "http://localhost:11434"
    runtime_settings.llm.ollama_model = "qwen3:1.7b"
    runtime_settings.llm.groq_api_key = ""

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.clear_cache", lambda: cleared.append("stt")
    )
    monkeypatch.setattr(
        "app.embeddings.clear_cache", lambda: cleared.append("emb")
    )

    us = user_settings.UserSettings(stt_mode="cloud", llm_mode="cloud")
    assert user_settings.sync_to_runtime(us) is False
    assert cleared == []

    us2 = user_settings.UserSettings(stt_mode="local", llm_mode="cloud")
    assert user_settings.sync_to_runtime(us2) is True
    assert cleared == ["stt", "emb"]


def test_sync_to_runtime_llm_mode_change_invalidates_embeddings_cache(monkeypatch):
    """Changing llm.mode alone must invalidate the embeddings provider cache —
    llm.mode is one half of the (stt.mode, llm.mode) key that gates embedding
    eligibility, so a stale Cloud embedding provider must not survive a switch
    to Local. STT is untouched, so the STT cache must NOT be cleared."""
    from app.core.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.engine = "auto"
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.initial_prompt = ""
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.mode = ProviderMode.CLOUD
    runtime_settings.llm.ollama_host = "http://localhost:11434"
    runtime_settings.llm.ollama_model = "qwen3:1.7b"
    runtime_settings.llm.groq_api_key = ""

    cleared: list[str] = []
    monkeypatch.setattr("app.stt.clear_cache", lambda: cleared.append("stt"))
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: cleared.append("emb"))

    us = user_settings.UserSettings(stt_mode="cloud", llm_mode="local")
    assert user_settings.sync_to_runtime(us) is False
    assert cleared == ["emb"]


def test_sync_to_runtime_propagates_initial_prompt_and_invalidates_cache(monkeypatch):
    """Changing the glossary mid-session must drop cached providers so the next
    transcribe call picks up the new value (cached providers freeze settings
    at construction time)."""
    from app.core.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.engine = "auto"
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.initial_prompt = ""
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.mode = ProviderMode.CLOUD
    runtime_settings.llm.ollama_host = "http://localhost:11434"
    runtime_settings.llm.ollama_model = "qwen3:1.7b"
    runtime_settings.llm.groq_api_key = ""

    cleared: list[str] = []
    monkeypatch.setattr("app.stt.clear_cache", lambda: cleared.append("stt"))
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: cleared.append("emb"))

    us = user_settings.UserSettings(initial_prompt="Tauri Pydantic")
    user_settings.sync_to_runtime(us)

    assert runtime_settings.stt.initial_prompt == "Tauri Pydantic"
    assert cleared == ["stt", "emb"]


def test_initial_prompt_max_length_validation():
    """The 500-char ceiling is enforced at Pydantic validation time."""
    too_long = "a" * 501
    with pytest.raises(ValueError):
        user_settings.UserSettings(initial_prompt=too_long)

    user_settings.UserSettings(initial_prompt="a" * 500)



def test_env_nested_stt_key_override(monkeypatch):
    """ENV override flows through STTSettings's own ``env_prefix="JUSTSAY_STT_"``
    on every fresh ``AppSettings()`` construction. ``Field(default_factory=...)``
    in ``core/config.py`` is what makes this true: the factory re-runs on each
    instance, picking up env mutations after module import.
    """
    monkeypatch.setenv("JUSTSAY_STT_GEMINI_API_KEY", "env-injected-key")
    fresh = AppSettings()
    assert fresh.stt.gemini_api_key == "env-injected-key"



def test_api_keys_round_trip(tmp_path):
    """gemini_api_key and groq_api_key persist through update_user_settings."""
    user_settings.update_user_settings({"gemini_api_key": "AIza-test", "groq_api_key": "gsk-test"})
    loaded = user_settings.get_user_settings()
    assert loaded.gemini_api_key == "AIza-test"
    assert loaded.groq_api_key == "gsk-test"


def test_a_settings_file_written_before_meeting_recording_still_loads(isolated):
    """Spec 074 AC: an existing settings.json has no meeting key, and must load
    with every pre-existing field intact and the new one defaulting to False."""
    import json

    settings_path = isolated["settings_dir"] / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "language": "en",
                "shortcut": "Ctrl+Alt+KeyQ",
                "stt_mode": "local",
                "ollama_model": "qwen3:1.7b",
            }
        ),
        encoding="utf-8",
    )
    user_settings._settings = None

    loaded = user_settings.get_user_settings()

    assert loaded.meeting_consent_acknowledged is False
    assert loaded.language == "en"
    assert loaded.shortcut == "Ctrl+Alt+KeyQ"
    assert loaded.stt_mode == "local"


def test_the_meeting_acknowledgement_round_trips_to_disk(isolated):
    """AC: the key is present in the file after the next save, and survives a
    reload — the disclosure must not reappear on every launch."""
    import json

    settings_path = isolated["settings_dir"] / "settings.json"

    user_settings.update_user_settings({"meeting_consent_acknowledged": True})

    assert json.loads(settings_path.read_text(encoding="utf-8"))[
        "meeting_consent_acknowledged"
    ] is True

    user_settings._settings = None
    assert user_settings.get_user_settings().meeting_consent_acknowledged is True


def test_the_meeting_acknowledgement_defaults_to_not_given():
    assert user_settings.UserSettings().meeting_consent_acknowledged is False


def test_sync_to_runtime_propagates_keys(monkeypatch):
    """sync_to_runtime pushes non-empty keys into runtime STT and LLM configs."""
    from app.core.config import settings as runtime_settings

    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.llm.groq_api_key = ""

    cleared: list[str] = []
    monkeypatch.setattr("app.stt.clear_cache", lambda: cleared.append("stt"))
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: cleared.append("emb"))

    us = user_settings.UserSettings(gemini_api_key="AIza-new", groq_api_key="gsk-new")
    user_settings.sync_to_runtime(us)

    assert runtime_settings.stt.gemini_api_key == "AIza-new"
    assert runtime_settings.stt.groq_api_key == "gsk-new"
    assert runtime_settings.llm.groq_api_key == "gsk-new"
    assert "stt" in cleared
    assert "emb" in cleared


def test_sync_to_runtime_preserves_env_key_when_user_key_empty(monkeypatch):
    """Empty UserSettings key must NOT overwrite a key already in the runtime (from .env)."""
    from app.core.config import settings as runtime_settings

    runtime_settings.stt.gemini_api_key = "env-key"
    runtime_settings.llm.groq_api_key = "env-groq"

    monkeypatch.setattr("app.stt.clear_cache", lambda: None)
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: None)

    us = user_settings.UserSettings(gemini_api_key="", groq_api_key="")
    user_settings.sync_to_runtime(us)

    assert runtime_settings.stt.gemini_api_key == "env-key"
    assert runtime_settings.llm.groq_api_key == "env-groq"


def _seed_history(directory: Path, count: int) -> None:
    """Create a real history.db at `directory` holding `count` entries."""
    directory.mkdir(parents=True, exist_ok=True)
    history.bootstrap(directory)
    for index in range(count):
        history.save_entry(text=f"entry {index}", duration_ms=100)
    with history._lock:
        history._close_conn_locked()


def _entry_count(db_path: Path) -> int:
    conn = history._connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    finally:
        conn.close()


def test_validate_output_dir_rejects_the_scratch_directory(isolated):
    scratch = isolated["settings_dir"] / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="temporary audio directory"):
        user_settings._validate_output_dir(str(scratch))


def test_validate_output_dir_rejects_a_directory_inside_scratch(isolated):
    nested = isolated["settings_dir"] / "tmp" / "history"
    nested.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="temporary audio directory"):
        user_settings._validate_output_dir(str(nested))


def test_validate_output_dir_accepts_the_root_that_contains_scratch(isolated):
    """The invariant is one-way (ADR 033). In the default layout the scratch
    directory sits INSIDE output_dir, so a symmetric no-nesting rule would
    reject every healthy install. Only the reverse is forbidden."""
    settings_dir = isolated["settings_dir"]
    (settings_dir / "tmp").mkdir(parents=True, exist_ok=True)

    assert user_settings._validate_output_dir(str(settings_dir)) == settings_dir.resolve()


def test_repair_leaves_a_healthy_output_dir_untouched(isolated, monkeypatch):
    settings_dir = isolated["settings_dir"]
    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(settings_dir))
    )

    assert user_settings.repair_scratch_output_dir() == settings_dir
    assert not list(settings_dir.glob("history.db.premigration-*"))


def test_repair_moves_history_out_of_the_scratch_directory(isolated, monkeypatch):
    settings_dir = isolated["settings_dir"]
    scratch = settings_dir / "tmp"
    _seed_history(scratch, 3)
    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(scratch))
    )

    resolved = user_settings.repair_scratch_output_dir()

    assert resolved == settings_dir
    assert _entry_count(settings_dir / "history.db") == 3
    assert user_settings.get_user_settings().output_dir == str(settings_dir)
    assert not (scratch / "history.db").exists()
    assert len(list(settings_dir.glob("history.db.premigration-*"))) == 1


def test_repair_never_lets_an_empty_target_displace_populated_history(isolated, monkeypatch):
    """The exact shape of the reported installation: history.db with real rows
    inside the scratch directory, and an EMPTY history.db already sitting at
    the destination. `relocate()` returns NEW_ALREADY_HAS_FILE here and adopts
    the empty file, which would hide every row -- consolidation merges instead.
    """
    settings_dir = isolated["settings_dir"]
    scratch = settings_dir / "tmp"
    _seed_history(scratch, 5)
    _seed_history(settings_dir, 0)
    assert _entry_count(settings_dir / "history.db") == 0

    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(scratch))
    )

    user_settings.repair_scratch_output_dir()

    assert _entry_count(settings_dir / "history.db") == 5


def test_repair_keeps_every_row_when_both_databases_have_entries(isolated, monkeypatch):
    settings_dir = isolated["settings_dir"]
    scratch = settings_dir / "tmp"
    _seed_history(settings_dir, 2)
    _seed_history(scratch, 3)

    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(scratch))
    )

    user_settings.repair_scratch_output_dir()

    assert _entry_count(settings_dir / "history.db") == 5


def test_repair_preserves_the_source_database_on_disk(isolated, monkeypatch):
    settings_dir = isolated["settings_dir"]
    scratch = settings_dir / "tmp"
    _seed_history(scratch, 4)
    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(scratch))
    )

    user_settings.repair_scratch_output_dir()

    kept = list(settings_dir.glob("history.db.premigration-*"))
    assert len(kept) == 1
    assert _entry_count(kept[0]) == 4


def test_repair_fires_when_output_dir_is_stored_non_canonically(isolated, monkeypatch):
    """settings.json holds whatever was written into it, not a normalised path.

    A redirected Windows profile or a stray `..` segment denotes the scratch
    directory without comparing equal to it, and the repair returning
    "healthy" there would leave real history inside the tree Clear Temp Files
    operates on -- failing silently in the one path that exists to prevent
    exactly that.
    """
    settings_dir = isolated["settings_dir"]
    scratch = settings_dir / "tmp"
    _seed_history(scratch, 6)

    non_canonical = settings_dir / ".." / settings_dir.name / "tmp"
    assert non_canonical != scratch
    assert non_canonical.resolve() == scratch.resolve()

    monkeypatch.setattr(
        user_settings, "_settings", user_settings.UserSettings(output_dir=str(non_canonical))
    )

    assert user_settings.repair_scratch_output_dir() == settings_dir
    assert _entry_count(settings_dir / "history.db") == 6
