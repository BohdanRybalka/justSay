"""Settings defaults — the persisted model and the runtime models must agree.

``UserSettings`` is what the user's ``settings.json`` holds; ``STTSettings``
and ``LLMSettings`` are what the pipeline reads at run time. Nine fields are
declared in both, three of them with a validation constraint declared in both,
and ``sync_to_runtime`` spells every name a third time. Nothing at import time
compares the copies, so editing two of the three places leaves the field
silently keeping its runtime default while the UI shows the new one.

Pinned rather than generated, for the reason ADR 045 gives for the
cross-language values: a single declaration would have to be imported across a
package boundary that ``test_import_layers.py`` already lists as a known cycle,
and the third spelling inside ``sync_to_runtime`` is a per-field assignment
with per-field semantics that no shared table removes.
"""

from __future__ import annotations

import pytest

from app.core.types import ProviderMode
from app.llm.config import LLMSettings
from app.preferences import user_settings
from app.preferences.user_settings import UserSettings
from app.stt.config import STTSettings

_STT = "stt"
_LLM = "llm"

_SHARED_FIELDS = [
    ("stt_engine", _STT, "engine"),
    ("whisper_model_size", _STT, "whisper_model_size"),
    ("whisper_device", _STT, "whisper_device"),
    ("initial_prompt", _STT, "initial_prompt"),
    ("cloud_routing_threshold", _STT, "cloud_routing_threshold"),
    ("gemini_api_key", _STT, "gemini_api_key"),
    ("groq_api_key", _STT, "groq_api_key"),
    ("ollama_host", _LLM, "ollama_host"),
    ("ollama_model", _LLM, "ollama_model"),
]

_RUNTIME_MODELS = {_STT: STTSettings, _LLM: LLMSettings}

_NON_DEFAULT_VALUES = {
    "stt_engine": "groq",
    "whisper_model_size": "small.en",
    "whisper_device": "cuda",
    "initial_prompt": "JustSay, Tauri, whisper.cpp",
    "cloud_routing_threshold": 12.5,
    "gemini_api_key": "AIza-sync-probe",
    "groq_api_key": "gsk-sync-probe",
    "ollama_host": "http://ollama.internal",
    "ollama_model": "llama3:8b",
}


@pytest.mark.parametrize(("stored_name", "package", "runtime_name"), _SHARED_FIELDS)
def test_the_stored_default_matches_the_runtime_default(stored_name, package, runtime_name):
    stored = UserSettings.model_fields[stored_name]
    runtime = _RUNTIME_MODELS[package].model_fields[runtime_name]

    assert stored.default == runtime.default


@pytest.mark.parametrize(("stored_name", "package", "runtime_name"), _SHARED_FIELDS)
def test_the_stored_constraints_match_the_runtime_constraints(stored_name, package, runtime_name):
    """A constraint on one side only is the shape the bug took.

    ``cloud_routing_threshold`` was ``30.0`` in both places with ``gt=0`` in
    the runtime copy alone, so the two disagreed about what counted as valid
    while looking identical in a diff.
    """
    stored = UserSettings.model_fields[stored_name]
    runtime = _RUNTIME_MODELS[package].model_fields[runtime_name]

    assert repr(stored.metadata) == repr(runtime.metadata)


def test_every_shared_field_reaches_the_runtime_object(monkeypatch):
    """``sync_to_runtime`` must not drop a field the two models both declare.

    This is the third spelling: a field can be added to both models and left
    out of the assignment block, and nothing else notices.
    """
    from app.core.config import settings as runtime_settings

    monkeypatch.setattr("app.stt.clear_cache", lambda: None)
    monkeypatch.setattr("app.embeddings.clear_cache", lambda: None)

    runtime_settings.stt.mode = ProviderMode.CLOUD
    runtime_settings.llm.mode = ProviderMode.CLOUD

    stored = UserSettings(**_NON_DEFAULT_VALUES)
    user_settings.sync_to_runtime(stored)

    for stored_name, package, runtime_name in _SHARED_FIELDS:
        landed = getattr(getattr(runtime_settings, package), runtime_name)
        assert landed == _NON_DEFAULT_VALUES[stored_name], stored_name


def test_the_shared_field_list_is_the_whole_overlap():
    """The table above is the pin, so it must not silently fall behind.

    A field added to both models under the same name but not to
    ``_SHARED_FIELDS`` would be unpinned while every test in this file still
    passed. A pair spelled differently on each side -- ``stt_engine`` against
    ``engine`` is the one that exists today -- is not caught here and cannot
    be, since nothing declares that the two names mean each other.
    """
    stored_names = set(UserSettings.model_fields)
    listed = {stored for stored, _, _ in _SHARED_FIELDS}

    overlap = stored_names & (set(STTSettings.model_fields) | set(LLMSettings.model_fields))

    assert overlap - listed == set()
