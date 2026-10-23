"""User settings — output_dir validation and runtime sync."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

from app.core.errors import ConfigurationError
from app.core.types import ProviderMode
from app.preferences import user_settings
from app.stt.config import STTSettings, stt_settings
from app.transcripts import history, relocation


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

    yield {"settings_dir": settings_dir}

    with history._lock:
        history._close_conn_locked()



def test_validate_output_dir_rejects_relative():
    with pytest.raises(ConfigurationError, match="absolute"):
        user_settings._validate_output_dir("relative/dir")


def test_validate_output_dir_rejects_empty():
    with pytest.raises(ConfigurationError, match="non-empty"):
        user_settings._validate_output_dir("   ")


def test_validate_output_dir_rejects_non_string():
    with pytest.raises(ConfigurationError, match="non-empty"):
        user_settings._validate_output_dir(42)


def test_validate_output_dir_rejects_when_path_is_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(ConfigurationError, match="not a directory"):
        user_settings._validate_output_dir(str(f))


def test_validate_output_dir_rejects_nonexistent_parent(tmp_path):
    target = tmp_path / "no" / "such" / "parent" / "dir"
    with pytest.raises(ConfigurationError, match="parent directory"):
        user_settings._validate_output_dir(str(target))


def test_validate_output_dir_rejects_when_not_writable(tmp_path, monkeypatch):
    target = tmp_path / "ro"
    target.mkdir()

    def deny_write(self, data):
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "write_bytes", deny_write)
    with pytest.raises(ConfigurationError, match="not writable"):
        user_settings._validate_output_dir(str(target))


def test_validate_output_dir_accepts_valid_dir(tmp_path):
    out = user_settings._validate_output_dir(str(tmp_path))
    assert out == tmp_path.resolve()


def test_validate_output_dir_rejects_forbidden_parent():
    """Names a directory that is genuinely a system root on the running host.

    `/etc/justsay` is the right probe on Linux and the wrong one on macOS,
    where `/etc` is a symlink and the interesting case is its own test below.
    """
    if sys.platform == "win32":
        forbidden = "C:/Windows/System32/justsay"
    elif sys.platform == "darwin":
        forbidden = "/Library/justsay"
    else:
        forbidden = "/etc/justsay"
    with pytest.raises(ConfigurationError, match="system directory"):
        user_settings._validate_output_dir(forbidden)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="/etc is a symlink to /private/etc on macOS only",
)
def test_validate_output_dir_rejects_etc_through_the_macos_symlink():
    """The candidate is resolved before comparison, so an unresolved `/etc`
    entry in `_FORBIDDEN_PARENTS` matches nothing and the guard silently stops
    firing -- only the OS permission bit refused the write, with an Errno 13
    the user cannot act on. See spec 156 and ADR 065."""
    with pytest.raises(ConfigurationError, match="system directory") as excinfo:
        user_settings._validate_output_dir("/etc/justsay")

    assert "/private/etc" in str(excinfo.value)


def test_forbidden_parents_are_stored_resolved():
    """`_validate_output_dir` resolves the candidate, so an entry that is not
    its own resolved form can never match one. This pins that relationship for
    every future entry instead of leaving it to whoever adds the next root."""
    for entry in user_settings._FORBIDDEN_PARENTS:
        assert entry == entry.resolve(strict=False), (
            f"{entry} is not its own resolved form; the candidate it should refuse "
            "resolves to a different path and the guard never fires"
        )



def test_update_output_dir_triggers_history_relocate(tmp_path, monkeypatch):
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    calls: list[Path] = []

    def fake_relocate(target: Path):
        calls.append(target)
        return relocation.RelocateOutcome.MOVED, None

    monkeypatch.setattr(relocation, "relocate", fake_relocate)

    user_settings.update_user_settings({"output_dir": str(new_dir)})
    assert len(calls) == 1
    assert calls[0] == new_dir.resolve()



def test_sync_to_runtime_releases_the_local_engine_only_on_a_mode_change(monkeypatch):
    from app.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.embeddings.ollama_host = "http://localhost:11434"

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr(
        "app.embeddings.clear_cache", lambda: cleared.append("emb")
    )

    us = user_settings.UserSettings(stt_mode="cloud")
    assert user_settings.sync_to_runtime(us) is False
    assert cleared == []

    us2 = user_settings.UserSettings(stt_mode="local")
    assert user_settings.sync_to_runtime(us2) is True
    assert cleared == ["stt", "emb"]


def test_sync_to_runtime_ollama_host_change_invalidates_embeddings_cache(monkeypatch):
    """Changing the Ollama host alone must invalidate the embeddings provider
    cache — a different host is a different Ollama, so a provider pinned to the
    old one must not survive. STT is untouched, so the local engine must NOT be
    released."""
    from app.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.initial_prompt = ""
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.embeddings.ollama_host = "http://localhost:11434"

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr(
        "app.embeddings.clear_cache", lambda: cleared.append("emb")
    )

    us = user_settings.UserSettings(stt_mode="cloud", ollama_host="http://127.0.0.1:11434")
    assert user_settings.sync_to_runtime(us) is False
    assert cleared == ["emb"]
    assert runtime_settings.embeddings.ollama_host == "http://127.0.0.1:11434"


def test_sync_to_runtime_propagates_initial_prompt_and_touches_no_engine(monkeypatch):
    """Every provider reads the glossary from the shared runtime settings on each
    request, so a changed glossary needs no engine restart and no cache clear."""
    from app.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.stt.initial_prompt = ""
    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""
    runtime_settings.embeddings.ollama_host = "http://localhost:11434"

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr(
        "app.embeddings.clear_cache", lambda: cleared.append("emb")
    )

    us = user_settings.UserSettings(initial_prompt="Tauri Pydantic")
    assert user_settings.sync_to_runtime(us) is False

    assert runtime_settings.stt.initial_prompt == "Tauri Pydantic"
    assert cleared == []


def test_sync_to_runtime_reports_a_model_or_device_change_and_leaves_the_engine_to_notice(
    monkeypatch,
):
    """The engine compares what it loaded with the settings and restarts itself once idle, so
    sync only reports the change (it gates the pre-warm) and releases nothing."""
    from app.config import settings as runtime_settings
    from app.core.types import ProviderMode

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: cleared.append("emb"))

    assert user_settings.sync_to_runtime(user_settings.UserSettings(whisper_model_size="small"))
    assert user_settings.sync_to_runtime(
        user_settings.UserSettings(whisper_model_size="small", whisper_device="cpu")
    )

    assert cleared == []


def test_initial_prompt_max_length_validation():
    """The 500-char ceiling is enforced at Pydantic validation time."""
    too_long = "a" * 501
    with pytest.raises(ValueError):
        user_settings.UserSettings(initial_prompt=too_long)

    user_settings.UserSettings(initial_prompt="a" * 500)



def test_a_freshly_built_stt_slice_reads_an_env_var_set_after_import(monkeypatch):
    """``env_prefix="JUSTSAY_STT_"`` is declared on ``STTSettings`` itself, so
    constructing the class directly reads whatever the environment holds at
    that moment rather than what it held when the module was first imported.
    """
    monkeypatch.setenv("JUSTSAY_STT_GEMINI_API_KEY", "env-injected-key")
    fresh = STTSettings()
    assert fresh.gemini_api_key == "env-injected-key"



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


def test_a_settings_file_written_before_the_llm_field_was_removed_still_loads(
    isolated, caplog
):
    """Spec 169 migration AC: a real pre-change file carries `llm_mode` and
    `ollama_model`, which no longer exist on the model. It must load silently —
    not through the salvage path, which would reset every other field to its
    default — keep every field the user can still act on, and drop the two dead
    keys from disk at the next save."""
    settings_path = isolated["settings_dir"] / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "language": "en",
                "shortcut": "Ctrl+Alt+KeyQ",
                "stt_mode": "local",
                "llm_mode": "cloud",
                "ollama_host": "http://127.0.0.1:11434",
                "ollama_model": "qwen3:1.7b",
                "gemini_api_key": "AIza-stored",
                "groq_api_key": "gsk-stored",
            }
        ),
        encoding="utf-8",
    )
    user_settings._settings = None

    with caplog.at_level(logging.WARNING, logger=user_settings.log.name):
        loaded = user_settings.get_user_settings()

    assert "Ignoring invalid stored settings" not in caplog.text
    assert loaded.language == "en"
    assert loaded.shortcut == "Ctrl+Alt+KeyQ"
    assert loaded.stt_mode == "local"
    assert loaded.ollama_host == "http://127.0.0.1:11434"
    assert loaded.gemini_api_key == "AIza-stored"
    assert loaded.groq_api_key == "gsk-stored"
    assert not hasattr(loaded, "llm_mode")
    assert not hasattr(loaded, "ollama_model")

    user_settings.update_user_settings({"language": "uk"})

    on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "llm_mode" not in on_disk
    assert "ollama_model" not in on_disk
    assert on_disk["ollama_host"] == "http://127.0.0.1:11434"
    assert on_disk["stt_mode"] == "local"


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


@pytest.mark.parametrize("switch", [True, False])
def test_a_file_from_the_record_meetings_switch_loads_and_drops_it(isolated, switch):
    settings_path = isolated["settings_dir"] / "settings.json"
    settings_path.write_text(
        json.dumps({"meeting_consent_acknowledged": True, "meetings_enabled": switch}),
        encoding="utf-8",
    )
    user_settings._settings = None

    assert user_settings.get_user_settings().meeting_consent_acknowledged is True
    user_settings.update_user_settings({"theme": "dark"})

    assert "meetings_enabled" not in json.loads(settings_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("stored", "meetings"), [({"language": "en"}, "en"), ({}, "uk")])
def test_a_file_without_a_meeting_language_starts_at_the_dictation_one(isolated, stored, meetings):
    settings_path = isolated["settings_dir"] / "settings.json"
    settings_path.write_text(json.dumps(stored), encoding="utf-8")
    user_settings._settings = None

    assert user_settings.get_user_settings().meetings_language == meetings


def test_a_chosen_meeting_language_reaches_the_meeting_jobs(isolated):
    user_settings.update_user_settings({"language": "en", "meetings_language": "auto"})

    user_settings._settings = None
    loaded = user_settings.get_user_settings()
    user_settings.sync_to_runtime(loaded)

    assert (loaded.language, loaded.meetings_language) == ("en", "auto")
    assert stt_settings.meetings_language == "auto"


@pytest.mark.parametrize(
    ("stored", "files"),
    [
        ({"stt_mode": "local"}, "local"),
        ({"stt_mode": "cloud"}, "cloud"),
        ({"stt_mode": "local", "files_engine": "cloud"}, "local"),
        ({"stt_mode": "cloud", "files_engine": "local"}, "local"),
        ({"stt_mode": "local", "files_engine": "local"}, "local"),
        ({}, "cloud"),
    ],
)
def test_a_stored_file_engine_is_kept_unless_it_sends_files_to_the_cloud_under_local_dictation(
    isolated, stored, files
):
    settings_path = isolated["settings_dir"] / "settings.json"
    settings_path.write_text(json.dumps(stored), encoding="utf-8")
    user_settings._settings = None

    assert user_settings.get_user_settings().files_engine == files


def test_files_follow_dictation_to_local_and_back(isolated):
    user_settings.update_user_settings({"stt_mode": "local"})
    assert user_settings.get_user_settings().files_engine == "local"

    user_settings.update_user_settings({"stt_mode": "cloud"})

    user_settings._settings = None
    loaded = user_settings.get_user_settings()
    assert (loaded.stt_mode, loaded.files_engine) == ("cloud", "cloud")


def test_a_dropped_file_goes_local_the_moment_dictation_does(isolated):
    user_settings.sync_to_runtime(user_settings.get_user_settings())
    assert stt_settings.files_mode == ProviderMode.CLOUD

    user_settings.update_user_settings({"stt_mode": "local"})
    user_settings.sync_to_runtime(user_settings.get_user_settings())

    assert stt_settings.files_mode == ProviderMode.LOCAL


def test_a_files_engine_chosen_apart_from_dictation_stays_when_dictation_changes(isolated):
    user_settings.update_user_settings({"files_engine": "local"})
    user_settings.update_user_settings({"stt_mode": "local"})
    user_settings.update_user_settings({"stt_mode": "cloud"})

    user_settings._settings = None
    loaded = user_settings.get_user_settings()
    assert (loaded.stt_mode, loaded.files_engine, loaded.files_follow_dictation) == (
        "cloud", "local", False,
    )


def test_choosing_the_dictation_engine_for_files_makes_them_follow_again(isolated):
    user_settings.update_user_settings({"files_engine": "local"})
    user_settings.update_user_settings({"files_engine": "cloud"})
    user_settings.update_user_settings({"stt_mode": "local"})

    assert user_settings.get_user_settings().files_engine == "local"


def test_a_files_engine_sent_with_a_dictation_change_is_judged_against_the_new_dictation(isolated):
    user_settings.update_user_settings({"stt_mode": "local", "files_engine": "cloud"})

    loaded = user_settings.get_user_settings()
    assert (loaded.files_engine, loaded.files_follow_dictation) == ("cloud", False)


def test_a_files_engine_differing_from_dictation_is_kept_when_chosen(isolated):
    user_settings.update_user_settings({"stt_mode": "local"})
    user_settings.update_user_settings({"files_engine": "cloud"})

    user_settings._settings = None
    loaded = user_settings.get_user_settings()
    assert (loaded.stt_mode, loaded.files_engine) == ("local", "cloud")


def test_files_are_detected_by_language_until_one_is_chosen(isolated):
    assert user_settings.UserSettings(language="en").files_language == "auto"


def test_a_chosen_files_engine_and_language_reach_the_file_jobs(isolated):
    user_settings.update_user_settings(
        {"language": "uk", "files_engine": "local", "files_language": "en"}
    )

    user_settings._settings = None
    loaded = user_settings.get_user_settings()
    user_settings.sync_to_runtime(loaded)

    assert (stt_settings.files_mode, stt_settings.files_language) == (ProviderMode.LOCAL, "en")
    assert stt_settings.mode == ProviderMode.CLOUD


def test_the_theme_follows_the_system_until_one_is_chosen():
    assert user_settings.UserSettings().theme == "system"


def test_a_chosen_theme_survives_a_restart(isolated):
    user_settings.update_user_settings({"theme": "dark"})

    user_settings._settings = None
    assert user_settings.get_user_settings().theme == "dark"


def test_an_unknown_theme_is_refused():
    with pytest.raises(ValueError):
        user_settings.update_user_settings({"theme": "sepia"})


def test_the_name_is_left_to_the_computer_until_one_is_chosen():
    assert user_settings.UserSettings().display_name == ""


def test_a_chosen_name_survives_a_restart(isolated):
    user_settings.update_user_settings({"display_name": "Богдан"})

    user_settings._settings = None
    assert user_settings.get_user_settings().display_name == "Богдан"


def test_a_name_longer_than_the_field_holds_is_refused():
    with pytest.raises(ValueError):
        user_settings.update_user_settings({"display_name": "x" * 81})


def test_dictated_text_is_pasted_where_you_type_until_turned_off():
    assert user_settings.UserSettings().paste_at_cursor is True


def test_turning_paste_off_survives_a_restart(isolated):
    user_settings.update_user_settings({"paste_at_cursor": False})

    user_settings._settings = None
    assert user_settings.get_user_settings().paste_at_cursor is False


def test_changing_the_language_remembers_the_one_before_it(isolated):
    user_settings.update_user_settings({"language": "uk"})
    user_settings.update_user_settings({"language": "en"})

    user_settings._settings = None
    assert user_settings.get_user_settings().previous_language == "uk"


def test_saving_the_same_language_keeps_the_one_before_it(isolated):
    user_settings.update_user_settings({"language": "uk"})
    user_settings.update_user_settings({"language": "en"})

    user_settings.update_user_settings({"language": "en", "theme": "dark"})

    assert user_settings.get_user_settings().previous_language == "uk"


def test_sync_to_runtime_propagates_keys(monkeypatch):
    """sync_to_runtime pushes non-empty keys into the runtime STT config."""
    from app.config import settings as runtime_settings

    runtime_settings.stt.gemini_api_key = ""
    runtime_settings.stt.groq_api_key = ""

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr("app.stt.routing.drop_cloud_provider", lambda: cleared.append("groq"))
    monkeypatch.setattr(
        "app.embeddings.clear_cache", lambda: cleared.append("emb")
    )

    us = user_settings.UserSettings(gemini_api_key="AIza-new", groq_api_key="gsk-new")
    assert user_settings.sync_to_runtime(us) is False

    assert runtime_settings.stt.gemini_api_key == "AIza-new"
    assert runtime_settings.stt.groq_api_key == "gsk-new"
    assert sorted(cleared) == ["emb", "groq"]


def test_an_empty_user_key_falls_back_to_the_environment_key(monkeypatch):
    """Removing a typed key brings back the one `.env` gave, not the typed one."""
    from app.config import settings as runtime_settings

    monkeypatch.setattr(
        user_settings,
        "_environment_keys",
        {"gemini_api_key": "env-key", "groq_api_key": "env-groq"},
    )
    runtime_settings.stt.gemini_api_key = "AIza-typed"
    runtime_settings.stt.groq_api_key = "gsk-typed"
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: None)

    user_settings.sync_to_runtime(user_settings.UserSettings(gemini_api_key="", groq_api_key=""))

    assert runtime_settings.stt.gemini_api_key == "env-key"
    assert runtime_settings.stt.groq_api_key == "env-groq"


def test_removing_a_typed_key_without_an_environment_key_stops_its_use(monkeypatch):
    """The runtime key empties and the Groq provider holding the old key is dropped."""
    from app.config import settings as runtime_settings

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.routing.release_local_if_unused", lambda settings: cleared.append("stt")
    )
    monkeypatch.setattr("app.stt.routing.drop_cloud_provider", lambda: cleared.append("groq"))
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: None)
    user_settings.sync_to_runtime(user_settings.UserSettings(groq_api_key="gsk-typed"))
    cleared.clear()

    changed_stt = user_settings.sync_to_runtime(user_settings.UserSettings(groq_api_key=""))

    assert runtime_settings.stt.groq_api_key == ""
    assert changed_stt is False
    assert cleared == ["groq"]


def test_the_environment_keys_are_read_from_the_environment_not_the_runtime(monkeypatch):
    """A key the user typed sits in the runtime settings and must not pass for an `.env` key."""
    from app.config import settings as runtime_settings

    monkeypatch.setenv("JUSTSAY_STT_GROQ_API_KEY", "env-groq")
    monkeypatch.setenv("JUSTSAY_STT__GEMINI_API_KEY", "env-gemini")
    runtime_settings.stt.groq_api_key = "gsk-typed"
    runtime_settings.stt.gemini_api_key = "AIza-typed"

    assert user_settings._keys_from_environment() == {
        "gemini_api_key": "env-gemini",
        "groq_api_key": "env-groq",
    }


def test_removing_only_the_google_key_drops_the_search_provider_holding_it(monkeypatch):
    """The Google key alone changing must still clear the embeddings cache."""
    from app.config import settings as runtime_settings

    cleared: list[str] = []
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: cleared.append("emb"))
    user_settings.sync_to_runtime(user_settings.UserSettings(gemini_api_key="AIza-typed"))
    cleared.clear()

    user_settings.sync_to_runtime(user_settings.UserSettings(gemini_api_key=""))

    assert runtime_settings.stt.gemini_api_key == ""
    assert cleared == ["emb"]


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

    with pytest.raises(ConfigurationError, match="temporary audio directory"):
        user_settings._validate_output_dir(str(scratch))


def test_validate_output_dir_rejects_a_directory_inside_scratch(isolated):
    nested = isolated["settings_dir"] / "tmp" / "history"
    nested.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ConfigurationError, match="temporary audio directory"):
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


def test_update_rejects_a_value_the_reload_would_refuse(isolated):
    """The write path enforces what the read path enforces (JS-94).

    ``model_copy(update=...)`` does not validate, so every ``Field``
    constraint on ``UserSettings`` used to be decorative on this path: a
    value could be accepted here and refused on the next load.
    """
    with pytest.raises(ValueError):
        user_settings.update_user_settings({"initial_prompt": "x" * 501})

    assert not (isolated["settings_dir"] / "settings.json").exists()


def test_a_rejected_stored_field_does_not_reset_the_other_settings(isolated, monkeypatch):
    """One bad value costs that field, not the whole file (JS-94).

    A single out-of-range entry used to send ``_load`` down its
    ``except ValueError: pass`` branch, which returned ``UserSettings()`` --
    discarding language, shortcut, output directory and both API keys with
    no message.
    """
    settings_dir = isolated["settings_dir"]
    user_settings.update_user_settings(
        {
            "language": "en",
            "shortcut": "Ctrl+Shift+KeyZ",
            "gemini_api_key": "AIza-real-key",
            "groq_api_key": "gsk-real-key",
        }
    )

    stored = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
    stored["initial_prompt"] = "x" * 5000
    stored["whisper_model_size"] = "../escape"
    (settings_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")

    monkeypatch.setattr(user_settings, "_settings", None)
    reloaded = user_settings.get_user_settings()

    assert reloaded.language == "en"
    assert reloaded.shortcut == "Ctrl+Shift+KeyZ"
    assert reloaded.gemini_api_key == "AIza-real-key"
    assert reloaded.groq_api_key == "gsk-real-key"
    assert reloaded.initial_prompt == ""
    assert reloaded.whisper_model_size == "large-v3-turbo"


def test_a_stored_engine_choice_from_before_groq_only_is_ignored_and_dropped(isolated, monkeypatch):
    """Settings written while Cloud had a Groq/Google choice still load: the
    two retired fields cost nothing else and vanish on the next save."""
    settings_dir = isolated["settings_dir"]
    user_settings.update_user_settings({"language": "en", "groq_api_key": "gsk-real-key"})
    stored = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
    stored["stt_engine"] = "gemini"
    stored["cloud_routing_threshold"] = 30.0
    (settings_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")

    monkeypatch.setattr(user_settings, "_settings", None)
    reloaded = user_settings.get_user_settings()
    assert reloaded.language == "en"
    assert reloaded.groq_api_key == "gsk-real-key"

    user_settings.update_user_settings({"language": "uk"})
    saved = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
    assert "stt_engine" not in saved
    assert "cloud_routing_threshold" not in saved


_CUT_MID_LETTER = '{"display_name": "Привіт"}'.encode()[:-3]


@pytest.mark.parametrize("stored", [b"{not json", _CUT_MID_LETTER, b"[]"])
def test_a_settings_file_that_holds_no_settings_is_kept_beside_the_defaults(
    isolated, monkeypatch, stored
):
    """Defaults take over, but the next save must not destroy the only copy of the keys."""
    settings_dir = isolated["settings_dir"]
    (settings_dir / "settings.json").write_bytes(stored)
    monkeypatch.setattr(user_settings, "_settings", None)

    assert user_settings.get_user_settings().language == "uk"
    user_settings.update_user_settings({"language": "en"})

    saved = json.loads((settings_dir / "settings.json").read_text(encoding="utf-8"))
    assert (settings_dir / "settings.json.bad").read_bytes() == stored
    assert saved["language"] == "en"


def test_a_file_that_cannot_be_set_aside_still_falls_back_to_defaults(
    isolated, monkeypatch, caplog
):
    (isolated["settings_dir"] / "settings.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(user_settings, "_settings", None)

    def refuse(source, target):
        raise PermissionError("held open by another program")

    monkeypatch.setattr(user_settings.os, "replace", refuse)

    with caplog.at_level(logging.WARNING, logger="app.preferences.user_settings"):
        assert user_settings.get_user_settings().language == "uk"

    assert "could not be set aside" in caplog.text


def test_a_crash_while_saving_leaves_the_previous_settings_whole(isolated, monkeypatch):
    """The new text sits in full in its own file before ``settings.json`` is touched."""
    settings_dir = isolated["settings_dir"]
    user_settings.update_user_settings({"groq_api_key": "gsk-real-key", "stt_mode": "local"})
    flushed_before_the_crash: list[dict] = []

    def power_lost(fd):
        unfinished = settings_dir / "settings.json.tmp"
        flushed_before_the_crash.append(json.loads(unfinished.read_text(encoding="utf-8")))
        raise OSError("power lost")

    monkeypatch.setattr(user_settings, "_force_to_disk", power_lost)
    with pytest.raises(OSError, match="power lost"):
        user_settings.update_user_settings({"language": "en", "groq_api_key": ""})

    assert flushed_before_the_crash[0]["language"] == "en"
    in_memory = user_settings.get_user_settings()
    monkeypatch.setattr(user_settings, "_settings", None)
    for kept in (in_memory, user_settings.get_user_settings()):
        assert (kept.groq_api_key, kept.stt_mode, kept.language) == ("gsk-real-key", "local", "uk")
    assert [path.name for path in settings_dir.glob("settings.json*")] == ["settings.json"]


def test_a_settings_file_that_cannot_be_read_falls_back_to_defaults_and_stays(
    isolated, monkeypatch, caplog
):
    settings_dir = isolated["settings_dir"]
    (settings_dir / "settings.json").write_text('{"language": "en"}', encoding="utf-8")
    monkeypatch.setattr(user_settings, "_settings", None)

    def locked(self):
        raise PermissionError("held open by another program")

    monkeypatch.setattr(Path, "read_bytes", locked)
    with caplog.at_level(logging.WARNING, logger="app.preferences.user_settings"):
        assert user_settings.get_user_settings().language == "uk"

    assert "Could not read" in caplog.text
    assert [path.name for path in settings_dir.glob("settings.json*")] == ["settings.json"]


def _record_the_flush(monkeypatch, full_fsync_refused: bool) -> list[tuple]:
    calls: list[tuple] = []

    def full_fsync(fd, command):
        if full_fsync_refused:
            raise OSError("not supported on this volume")
        calls.append(("F_FULLFSYNC", fd, command))

    fake_fcntl = types.SimpleNamespace(F_FULLFSYNC=51, fcntl=full_fsync)
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)
    monkeypatch.setattr(user_settings.sys, "platform", "darwin")
    monkeypatch.setattr(user_settings.os, "fsync", lambda fd: calls.append(("fsync", fd)))
    return calls


def test_on_a_mac_saved_settings_are_pushed_past_the_drive_cache(monkeypatch):
    calls = _record_the_flush(monkeypatch, full_fsync_refused=False)

    user_settings._force_to_disk(7)

    assert calls == [("F_FULLFSYNC", 7, 51)]


def test_a_mac_volume_that_refuses_a_full_flush_still_gets_fsync(monkeypatch):
    calls = _record_the_flush(monkeypatch, full_fsync_refused=True)

    user_settings._force_to_disk(7)

    assert calls == [("fsync", 7)]


def test_falling_back_to_defaults_is_never_silent(isolated, monkeypatch, caplog):
    """Every path that discards stored settings says so (JS-94).

    The defect this file's other cases cover was a silent discard. Widening
    the read to catch ``OSError`` added a second silent one, so both now log.
    """
    (isolated["settings_dir"] / "settings.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(user_settings, "_settings", None)

    with caplog.at_level(logging.WARNING, logger="app.preferences.user_settings"):
        assert user_settings.get_user_settings().language == "uk"

    assert "starting from defaults" in caplog.text


@pytest.mark.parametrize(
    "host",
    [
        "http://evil.example:11434",
        "http://10.0.0.5:11434",
        "http://localhost@evil.example",
        "ftp://localhost",
        "",
    ],
)
def test_an_ollama_host_off_this_computer_is_refused_and_nothing_changes(host):
    before = user_settings.get_user_settings().ollama_host

    with pytest.raises(ConfigurationError, match="on this computer"):
        user_settings.update_user_settings({"ollama_host": host})

    assert user_settings.get_user_settings().ollama_host == before


@pytest.mark.parametrize("host", ["https://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_an_ollama_host_on_this_computer_is_accepted(host):
    result = user_settings.update_user_settings({"ollama_host": host})

    assert result.settings.ollama_host == host


def test_a_stored_ollama_host_off_this_computer_loads_as_the_default_and_keeps_the_rest(isolated):
    (isolated["settings_dir"] / "settings.json").write_text(
        json.dumps({"language": "en", "ollama_host": "http://evil.example:11434"}),
        encoding="utf-8",
    )
    user_settings._settings = None

    loaded = user_settings.get_user_settings()

    assert loaded.ollama_host == "http://localhost:11434"
    assert loaded.language == "en"


def test_a_refused_ollama_host_stops_the_request_before_the_history_moves(tmp_path):
    before = user_settings.get_user_settings().output_dir

    with pytest.raises(ConfigurationError, match="on this computer"):
        user_settings.update_user_settings(
            {"output_dir": str(tmp_path / "elsewhere"), "ollama_host": "http://evil.example"}
        )

    assert user_settings.get_user_settings().output_dir == before
    assert not (tmp_path / "elsewhere").exists()
