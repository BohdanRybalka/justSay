"""User settings — output_dir validation and runtime sync."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.core import history, user_settings
from app.core.config import AppSettings


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    settings_dir = home / ".justsay"
    settings_dir.mkdir()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(user_settings, "SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(user_settings, "SETTINGS_PATH", settings_dir / "settings.json")
    monkeypatch.setattr(user_settings, "_settings", None)
    monkeypatch.setattr(history, "_output_dir", settings_dir)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    yield {"home": home, "settings_dir": settings_dir}

    with history._lock:
        history._close_conn_locked()


# --- _validate_output_dir branches --------------------------------------

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

    def deny_write(self, data):  # patched method
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


# --- update_user_settings → history.relocate -----------------------------

def test_update_output_dir_triggers_history_relocate(tmp_path, monkeypatch):
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    calls: list[Path] = []

    def fake_relocate(target: Path):
        calls.append(target)
        return history.RelocateResult.MOVED, None

    monkeypatch.setattr(history, "relocate", fake_relocate)

    user_settings.update_user_settings({"output_dir": str(new_dir)})
    assert len(calls) == 1
    assert calls[0] == new_dir.resolve()


# --- sync_to_runtime ----------------------------------------------------

def test_sync_to_runtime_clears_stt_cache_only_on_change(monkeypatch):
    from app.core.config import settings as runtime_settings
    from app.core.types import ProviderMode

    # Pre-align runtime to match defaults — so the only differences we test are
    # the ones we deliberately introduce below.
    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.stt.engine = "auto"
    runtime_settings.stt.whisper_model_size = "large-v3-turbo"
    runtime_settings.stt.whisper_device = "auto"
    runtime_settings.llm.mode = ProviderMode.CLOUD
    runtime_settings.llm.ollama_host = "http://localhost:11434"
    runtime_settings.llm.ollama_model = "qwen3:1.7b"

    cleared: list[str] = []
    monkeypatch.setattr(
        "app.stt.clear_cache", lambda: cleared.append("stt")
    )
    monkeypatch.setattr(
        "app.llm.clear_cache", lambda: cleared.append("llm")
    )

    # No diffs → no clear.
    us = user_settings.UserSettings(stt_mode="cloud", llm_mode="cloud")
    user_settings.sync_to_runtime(us)
    assert cleared == []

    # STT mode change → STT clear fires.
    us2 = user_settings.UserSettings(stt_mode="local", llm_mode="cloud")
    user_settings.sync_to_runtime(us2)
    assert cleared == ["stt"]


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
    runtime_settings.llm.mode = ProviderMode.CLOUD
    runtime_settings.llm.ollama_host = "http://localhost:11434"
    runtime_settings.llm.ollama_model = "qwen3:1.7b"

    cleared: list[str] = []
    monkeypatch.setattr("app.stt.clear_cache", lambda: cleared.append("stt"))
    monkeypatch.setattr("app.llm.clear_cache", lambda: cleared.append("llm"))

    us = user_settings.UserSettings(initial_prompt="Tauri Pydantic")
    user_settings.sync_to_runtime(us)

    assert runtime_settings.stt.initial_prompt == "Tauri Pydantic"
    assert cleared == ["stt"]


def test_initial_prompt_max_length_validation():
    """The 500-char ceiling is enforced at Pydantic validation time."""
    too_long = "a" * 501
    with pytest.raises(ValueError):
        user_settings.UserSettings(initial_prompt=too_long)

    # 500 chars exactly is fine.
    user_settings.UserSettings(initial_prompt="a" * 500)


# --- ENV nested override after Phase 1.4 default refactor ----------------

def test_env_nested_stt_key_override(monkeypatch):
    """ENV override flows through STTSettings's own ``env_prefix="JUSTSAY_STT_"``
    on every fresh ``AppSettings()`` construction. ``Field(default_factory=...)``
    in ``core/config.py`` is what makes this true: the factory re-runs on each
    instance, picking up env mutations after module import.
    """
    monkeypatch.setenv("JUSTSAY_STT_GEMINI_API_KEY", "env-injected-key")
    fresh = AppSettings()
    assert fresh.stt.gemini_api_key == "env-injected-key"
