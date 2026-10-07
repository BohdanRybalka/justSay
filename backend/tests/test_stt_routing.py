from unittest.mock import patch

import pytest

from app.core.types import ProviderMode
from app.pipeline import service
from app.stt.config import STTSettings
from app.stt.groq_whisper import GroqWhisperSTTProvider
from app.stt.local import LocalSTTProvider
from app.stt.local_whisper_cpp import WhisperCppServerSTTProvider
from app.stt.routing import _providers, clear_cache, get_provider


@pytest.fixture(autouse=True)
def _clear_stt_cache():
    clear_cache()
    yield
    clear_cache()


def _cloud_settings(**overrides) -> STTSettings:
    defaults = dict(mode=ProviderMode.CLOUD, gemini_api_key="g", groq_api_key="q")
    defaults.update(overrides)
    return STTSettings(**defaults)


def test_local_mode_returns_local():
    """Proves only that LOCAL does not reach a cloud provider — see the marked
    test below for the platform-routing pin (JS-97)."""
    s = STTSettings(mode=ProviderMode.LOCAL)
    assert isinstance(get_provider(s.mode, s), LocalSTTProvider)


def test_local_mode_constructs_no_cloud_provider_even_with_both_keys_set():
    """The zero-leak guarantee at the Local/Cloud decision point itself: the
    returned provider declares ``is_local`` and the cache holds no cloud client,
    so no API key was read."""
    s = STTSettings(mode=ProviderMode.LOCAL, gemini_api_key="g", groq_api_key="q")
    provider = get_provider(s.mode, s)
    assert provider.is_local is True
    assert GroqWhisperSTTProvider not in _providers


@pytest.mark.asyncio
async def test_the_mode_endpoint_routes_to_local_and_the_switch_back_reaches_cloud(
    client, monkeypatch
):
    """`PUT /stt/mode` and the binding the pipeline holds read one object, in
    both directions. Routing is asked through ``service.stt_settings`` — the
    module attribute ``process_audio`` itself passes — so an endpoint writing
    onto one settings object while the pipeline reads another reddens this test.
    """
    constructed: list[str] = []
    original = GroqWhisperSTTProvider.__init__

    def _record(self, *args, **kwargs):
        constructed.append(type(self).__name__)
        original(self, *args, **kwargs)

    monkeypatch.setattr(GroqWhisperSTTProvider, "__init__", _record)

    switched_local = await client.put("/stt/mode", json={"mode": "local"})
    assert switched_local.status_code == 200
    assert switched_local.json()["stt_mode"] == ProviderMode.LOCAL.value

    local_provider = get_provider(service.stt_settings.mode, service.stt_settings)
    assert local_provider.is_local is True, (
        f"the mode endpoint wrote local but routing returned {type(local_provider).__name__}"
    )
    assert constructed == [], f"Local mode constructed {constructed}"
    assert GroqWhisperSTTProvider not in _providers

    switched_cloud = await client.put("/stt/mode", json={"mode": "cloud"})
    assert switched_cloud.status_code == 200
    assert switched_cloud.json()["stt_mode"] == ProviderMode.CLOUD.value
    assert switched_cloud.json()["model"] == "groq/whisper-large-v3-turbo"

    cloud_provider = get_provider(service.stt_settings.mode, service.stt_settings)
    assert isinstance(cloud_provider, GroqWhisperSTTProvider)


@pytest.mark.no_factory_stub
def test_local_mode_routes_to_this_platforms_local_provider(monkeypatch):
    """The unmarked siblings cannot say this: the autouse
    `_force_faster_whisper_for_local` fixture pins the class they assert."""
    monkeypatch.setattr(
        "app.stt.local_factory.get_local_provider_class",
        lambda: WhisperCppServerSTTProvider,
    )
    clear_cache()
    s = STTSettings(mode=ProviderMode.LOCAL)
    assert type(get_provider(s.mode, s)) is WhisperCppServerSTTProvider


def test_cloud_mode_transcribes_on_groq_even_without_a_groq_key():
    """Cloud has one engine. A Google key alone does not reroute: the Groq
    provider is returned and raises its own key-missing message on use."""
    s = _cloud_settings(groq_api_key="")
    assert isinstance(get_provider(s.mode, s), GroqWhisperSTTProvider)


def test_same_provider_is_cached_across_calls():
    s = _cloud_settings()
    assert get_provider(s.mode, s) is get_provider(s.mode, s)


def test_the_status_reads_raise_on_a_provider_that_declares_neither_member():
    """ADR 075's second half, pinned: a missing member must raise, not answer.

    `get_local_load_error()` and `is_model_loaded()` read `last_load_error`
    and `is_loaded` straight off the cached local provider. Read through a
    defaulting `getattr` instead, a local provider class that spells either
    name differently gets "no error" and "not loaded" answered on its behalf:
    `GET /stt/local/status` draws a healthy indicator and reports a model that
    never loads, for the life of the process, which is the defect ADR 075
    exists to end rather than to relocate.

    Nothing else in the suite reaches this. Every class
    `get_local_provider_class()` can return today declares both members, so
    restoring the two defaults leaves every other test green -- which is why
    the object here is planted in the cache rather than resolved from the
    factory. The class it stands for is the one the factory does not have yet.

    The factory is imported in the function body, not at module level: the
    autouse `_force_faster_whisper_for_local` fixture patches it on
    `app.stt.local_factory` itself, and only a body-local import reads the
    same object the two functions under test resolve.
    """
    from app.stt.local_factory import get_local_provider_class
    from app.stt.routing import get_local_load_error, is_model_loaded

    class _ProviderDeclaringNothing:
        pass

    _providers[get_local_provider_class()] = _ProviderDeclaringNothing()

    with pytest.raises(AttributeError, match="last_load_error"):
        get_local_load_error(STTSettings(mode=ProviderMode.LOCAL))

    with pytest.raises(AttributeError, match="is_loaded"):
        is_model_loaded()


def test_clear_cache_triggers_cleanup_on_all():
    cloud = _cloud_settings()
    local = STTSettings(mode=ProviderMode.LOCAL)
    groq = get_provider(cloud.mode, cloud)
    whisper = get_provider(local.mode, local)

    with patch.object(groq, "cleanup") as groq_mock, patch.object(whisper, "cleanup") as local_mock:
        clear_cache()
        groq_mock.assert_called_once()
        local_mock.assert_called_once()


def test_detect_duration_returns_none_for_missing_file(tmp_path):
    from app.pipeline.utils import detect_duration

    assert detect_duration(tmp_path / "no-such-file.wav") is None


def test_detect_duration_reads_real_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from app.pipeline.utils import detect_duration

    path = tmp_path / "two-seconds.wav"
    samples = np.zeros(32000, dtype=np.float32)
    sf.write(str(path), samples, 16000)

    duration = detect_duration(path)
    assert duration is not None
    assert 1.9 < duration < 2.1
