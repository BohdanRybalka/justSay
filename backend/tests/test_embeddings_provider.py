"""Embedding provider selection — privacy eligibility (specs 003, 169).

Single most important AC: eligibility must be derived STRICTLY from
``stt.mode`` with no cloud bypass hiding inside the local branch. Mocks the
factory's internal constructors directly so a direct cloud-SDK bypass cannot
pass.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import socket
import sys
import threading
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.constants import GEMINI_EMBEDDING_TIMEOUT_SECONDS, GEMINI_TIMEOUT_SECONDS
from app.core.errors import ConfigurationError
from app.core.types import ProviderMode
from app.embeddings import LOCAL_MISSING_MODEL_REASON, clear_cache, resolve_embedding_provider
from app.embeddings.cloud import CloudEmbeddingProvider
from app.embeddings.config import EmbeddingSettings
from app.embeddings.local import LocalEmbeddingProvider
from app.stt.config import STTSettings
from tests.conftest import holding_no_frames


@pytest.fixture(autouse=True)
def _clear_embeddings_cache():
    clear_cache()
    yield
    clear_cache()


def _settings(stt_mode: ProviderMode):
    stt = STTSettings(mode=stt_mode, gemini_api_key="key")
    emb = EmbeddingSettings()
    return stt, emb


@pytest.mark.asyncio
@pytest.mark.parametrize("stt_mode", list(ProviderMode))
async def test_eligibility_matrix(stt_mode):
    """resolve_embedding_provider returns a CloudEmbeddingProvider for CLOUD and
    a LocalEmbeddingProvider for LOCAL with Ollama reporting nomic-embed-text
    installed. Those are the only two outcomes: with one switch there is no
    third state, so the parametrisation covers every value of ProviderMode and
    each one must yield a provider.
    """
    clear_cache()
    stt, emb = _settings(stt_mode)

    fake_cloud = MagicMock(name="CloudEmbeddingProvider-instance")
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.cloud.CloudEmbeddingProvider", return_value=fake_cloud) as cloud_ctor,
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local) as local_ctor,
        patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)) as avail,
    ):
        provider, reason = await resolve_embedding_provider(stt, emb)

    assert reason is None
    if stt_mode is ProviderMode.CLOUD:
        assert provider is fake_cloud
        cloud_ctor.assert_called_once()
        local_ctor.assert_not_called()
        avail.assert_not_called()
    else:
        assert provider is fake_local
        avail.assert_called_once()
        local_ctor.assert_called_once()
        cloud_ctor.assert_not_called()


@pytest.mark.asyncio
async def test_local_mode_disabled_when_model_not_pulled():
    """LOCAL with Ollama NOT reporting nomic-embed-text -> None + a specific
    actionable reason, and LocalEmbeddingProvider is never constructed (no
    half-built client left hanging around)."""
    stt, emb = _settings(ProviderMode.LOCAL)

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider") as local_ctor,
        patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=False)),
    ):
        provider, reason = await resolve_embedding_provider(stt, emb)

    assert provider is None
    local_ctor.assert_not_called()
    assert reason is not None
    assert "nomic-embed-text" in reason


@pytest.mark.asyncio
async def test_local_mode_never_constructs_the_cloud_provider(tmp_path, monkeypatch):
    """The zero-leak AC, driven from a real settings file rather than a
    hand-built STTSettings: a file written before this change still carries
    ``llm_mode: "cloud"``, which used to be half the eligibility key. Loading it
    and syncing it to the runtime must leave a Local-mode install on the local
    branch, with CloudEmbeddingProvider never constructed — and it is the
    constructor that is patched, so no lazily-built Gemini client can exist
    either.
    """
    import json

    from app.core.config import settings as runtime_settings
    from app.preferences import user_settings

    settings_dir = tmp_path / ".justsay"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(
        json.dumps({"stt_mode": "local", "llm_mode": "cloud", "gemini_api_key": "AIza-stored"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("JUSTSAY_DATA_DIR", str(settings_dir))
    monkeypatch.setattr(user_settings, "_settings", None)
    monkeypatch.setattr("app.stt.routing.clear_cache", lambda: None)

    saved_mode = runtime_settings.stt.mode
    try:
        user_settings.sync_to_runtime(user_settings.get_user_settings())
        assert runtime_settings.stt.mode is ProviderMode.LOCAL

        clear_cache()
        with (
            patch("app.embeddings.cloud.CloudEmbeddingProvider") as cloud_ctor,
            patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)),
            patch("app.embeddings.local.LocalEmbeddingProvider") as local_ctor,
        ):
            provider, reason = await resolve_embedding_provider(
                runtime_settings.stt, runtime_settings.embeddings
            )
    finally:
        runtime_settings.stt.mode = saved_mode

    cloud_ctor.assert_not_called()
    local_ctor.assert_called_once()
    assert provider is local_ctor.return_value
    assert reason is None


@pytest.mark.asyncio
async def test_factory_caches_provider_by_mode():
    stt, emb = _settings(ProviderMode.CLOUD)
    with patch("app.embeddings.cloud.CloudEmbeddingProvider", return_value=MagicMock()):
        p1, _ = await resolve_embedding_provider(stt, emb)
        p2, _ = await resolve_embedding_provider(stt, emb)
    assert p1 is p2


@pytest.mark.asyncio
async def test_clear_cache_forces_reresolve():
    stt, emb = _settings(ProviderMode.CLOUD)
    with patch(
        "app.embeddings.cloud.CloudEmbeddingProvider", side_effect=lambda **_: MagicMock()
    ) as ctor:
        await resolve_embedding_provider(stt, emb)
        clear_cache()
        await resolve_embedding_provider(stt, emb)
    assert ctor.call_count == 2


@pytest.mark.asyncio
async def test_stale_unavailable_cache_reprobes_and_flips_available():
    """A cached LOCAL negative result caused by a missing model
    must not be served verbatim forever — it must re-probe Ollama on every
    call until the model appears, then cache the resulting provider
    normally, without an intervening clear_cache()."""
    stt, emb = _settings(ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available",
            new=AsyncMock(side_effect=[False, True]),
        ) as avail,
    ):
        provider1, reason1 = await resolve_embedding_provider(stt, emb)
        provider2, reason2 = await resolve_embedding_provider(stt, emb)

    assert (provider1, reason1) == (None, LOCAL_MISSING_MODEL_REASON)
    assert provider2 is fake_local
    assert reason2 is None
    assert avail.call_count == 2


@pytest.mark.asyncio
async def test_available_cache_reprobes_and_flips_to_unavailable():
    """A cached positive LOCAL result must not be served verbatim
    forever either — it re-probes on every call same as the negative branch,
    and flips to LOCAL_MISSING_MODEL_REASON (cleaning up the now-stale
    provider) when the model disappears (e.g. `ollama rm nomic-embed-text`).
    This supersedes the old "does not reprobe" assertion on purpose — that
    was exactly the gap this item closes, not a regression."""
    stt, emb = _settings(ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)
        ) as avail,
    ):
        provider1, reason1 = await resolve_embedding_provider(stt, emb)
        assert avail.call_count == 1

        avail.return_value = False
        provider2, reason2 = await resolve_embedding_provider(stt, emb)

    assert provider1 is fake_local
    assert reason1 is None
    assert provider2 is None
    assert reason2 == LOCAL_MISSING_MODEL_REASON
    assert avail.call_count == 2
    fake_local.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_available_cache_reuses_instance_while_still_available():
    """While the model stays available across consecutive calls, the same
    LocalEmbeddingProvider instance is reused (not reconstructed) — but
    each call still re-probes is_model_available()."""
    stt, emb = _settings(ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch(
            "app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local
        ) as local_ctor,
        patch(
            "app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)
        ) as avail,
    ):
        provider1, _ = await resolve_embedding_provider(stt, emb)
        provider2, _ = await resolve_embedding_provider(stt, emb)

    assert provider1 is provider2 is fake_local
    local_ctor.assert_called_once()
    assert avail.call_count == 2
    fake_local.cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_local_calls_serialize_and_stay_consistent():
    """Two concurrent LOCAL calls must never interleave their
    probe/decide/cleanup-or-reuse/cache-write sequence. A hand-written
    async probe (not a plain AsyncMock) tracks how many calls are
    mid-probe at once via a shared counter incremented on entry and
    decremented on exit, with a real `await asyncio.sleep(0)` yield point
    in between — if `_local_reprobe_lock`'s scope were wrong, both
    coroutines could be mid-probe simultaneously and the counter would
    observe 2. The two calls' probes return different results (True then
    False, by call order), which under correct serialization means the
    first call resolves to a fresh provider and the second — observing
    that committed result — flips it to unavailable and cleans it up
    exactly once."""
    stt, emb = _settings(ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    in_flight = 0
    max_in_flight = 0
    call_count = 0
    results = [True, False]

    async def instrumented_is_model_available(*args, **kwargs):
        nonlocal in_flight, max_in_flight, call_count
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        assert in_flight == 1, "a second probe started while one was still mid-flight"
        result = results[call_count]
        call_count += 1
        in_flight -= 1
        return result

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available",
            new=instrumented_is_model_available,
        ),
    ):
        result1, result2 = await asyncio.gather(
            resolve_embedding_provider(stt, emb),
            resolve_embedding_provider(stt, emb),
        )

    assert max_in_flight == 1
    assert {result1, result2} == {(fake_local, None), (None, LOCAL_MISSING_MODEL_REASON)}
    fake_local.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_local_calls_each_reprobe_no_coalescing():
    """Serializing LOCAL resolutions behind `_local_reprobe_lock`
    must not accidentally coalesce concurrent callers into a single probe:
    3 concurrent calls still make 3 independent `is_model_available()`
    round-trips (Spec 006/008's no-coalescing design, unchanged), while
    still reusing the same LocalEmbeddingProvider instance across all of
    them, now proven under real concurrency rather than just sequentially."""
    stt, emb = _settings(ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)
        ) as avail,
    ):
        results = await asyncio.gather(
            resolve_embedding_provider(stt, emb),
            resolve_embedding_provider(stt, emb),
            resolve_embedding_provider(stt, emb),
        )

    assert avail.call_count == 3
    assert all(provider is fake_local for provider, _ in results)
    assert all(reason is None for _, reason in results)



@pytest.mark.asyncio
async def test_embed_entry_background_noop_when_disabled():
    """When resolve_embedding_provider returns None, embed_entry_background
    must be a no-op that never imports/instantiates CloudEmbeddingProvider —
    closes the 'silent fallback to cloud' failure mode explicitly, not just
    by absence of a code path."""
    from app.transcripts import history, vector_store

    with (
        patch.object(history, "_vec_available", True),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(None, "disabled")),
        ),
        patch("app.embeddings.cloud.CloudEmbeddingProvider") as cloud_ctor,
    ):
        await vector_store.embed_entry_background("entry-id", "some text")

    cloud_ctor.assert_not_called()


@pytest.mark.asyncio
async def test_embed_entry_background_noop_when_vec_unavailable():
    """If the sqlite-vec extension failed to load, embed_entry_background
    must not even attempt to resolve a provider."""
    from app.transcripts import history, vector_store

    with (
        patch.object(history, "_vec_available", False),
        patch("app.embeddings.resolve_embedding_provider") as resolve_mock,
    ):
        await vector_store.embed_entry_background("entry-id", "some text")

    resolve_mock.assert_not_called()




def test_cloud_embedding_model_name():
    provider = CloudEmbeddingProvider(gemini_api_key="key", model="text-embedding-004")
    assert provider.model_name == "gemini/text-embedding-004"


def test_cloud_embedding_requires_api_key():
    """A missing key is something the user fixes in Settings, so it is a refusal.

    The class is what every caller of `embed()` sees: `words.py` and
    `vector_store.embed_entry_background` both catch `Exception` and degrade
    silently, so the migration changes no behaviour here — it makes the reason
    readable to anything that ever stops swallowing it.
    """
    provider = CloudEmbeddingProvider(gemini_api_key="", model="text-embedding-004")
    with pytest.raises(ConfigurationError, match="missing"):
        provider._get_client()


@pytest.mark.asyncio
async def test_cloud_embedding_embed_parses_response():
    """Exercises the real `_call_embed` parsing logic
    (`response.embeddings[0].values`), not a mocked-away static method."""
    provider = CloudEmbeddingProvider(gemini_api_key="test-key", model="text-embedding-004")

    fake_embedding = MagicMock()
    fake_embedding.values = [0.1, 0.2, 0.3]
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]

    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = fake_response
    provider._client = fake_client

    result = await provider.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    fake_client.models.embed_content.assert_called_once_with(
        model="text-embedding-004", contents="hello world"
    )


_UNANSWERED_EMBED_TIMEOUT_MS = 300


def test_cloud_embedding_client_carries_a_timeout_in_milliseconds():
    """AC: the embedding client is bounded the way the STT client is.

    `embed()` runs under `asyncio.to_thread`, which cannot be cancelled, and it
    is reached from `/history/search` and from the background indexer. An
    unanswered embed therefore parks a default-executor worker for the life of
    the process and the search request never returns -- the same defect the
    STT client was fixed for, on a second live path.

    The budget is its own constant rather than the STT one. 300 s is sized for
    uploading a recording of up to `MAX_UPLOAD_SIZE`; an embedding sends one
    short string, so borrowing that number lets a handful of unanswered embeds
    hold the shared default executor for five minutes each.

    `HttpOptions.timeout` is milliseconds, so the assertion is on the scaled
    number: passing seconds would give a 30 ms budget and break every cloud
    embedding.
    """
    from tests.conftest import fake_genai_modules

    provider = CloudEmbeddingProvider(gemini_api_key="test-key", model="text-embedding-004")
    client_class = MagicMock()

    with patch.dict(sys.modules, fake_genai_modules(client_class)):
        provider._get_client()

    http_options = client_class.call_args.kwargs["http_options"]
    assert http_options.timeout == int(GEMINI_EMBEDDING_TIMEOUT_SECONDS * 1000)
    assert GEMINI_EMBEDDING_TIMEOUT_SECONDS == 30.0
    assert GEMINI_EMBEDDING_TIMEOUT_SECONDS < GEMINI_TIMEOUT_SECONDS, (
        "an embedding is one short string, so it must not inherit the budget "
        "sized for uploading a whole recording"
    )


def test_an_embedding_request_that_is_never_answered_raises_a_timeout():
    """AC: an unanswered embedding call ends instead of holding a worker.

    The socket is bound and listening but never accepted, so the kernel
    completes the handshake out of the backlog and the request then waits on a
    response that never comes -- the shape of the hang this bounds, which a
    refused connection would not reproduce. The call runs on a worker joined
    with a hard cap, so dropping `http_options` fails this test on
    `is_alive()` instead of hanging the suite.
    """
    pytest.importorskip(
        "google.genai",
        reason="the real SDK is what carries the timeout to httpx; it lives in the "
        "optional cloud extra",
    )
    import httpx
    from google import genai
    from google.genai import types

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    caught: list[BaseException] = []
    client_ref: list = []

    def _call() -> None:
        with genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(
                base_url=f"http://127.0.0.1:{port}",
                timeout=_UNANSWERED_EMBED_TIMEOUT_MS,
            ),
        ) as client:
            client_ref.append(weakref.ref(client))
            try:
                CloudEmbeddingProvider._call_embed(client, "text-embedding-004", "hello")
            except BaseException as e:
                caught.append(holding_no_frames(e))

    worker = threading.Thread(target=_call, name="embed-timeout-probe", daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    finished = not worker.is_alive()
    listener.close()

    assert finished, (
        "the embedding call was still waiting after 5 s, so the client carries no "
        "timeout and a history search against an unanswering endpoint never returns"
    )
    assert caught, "the call returned a result from a server that never answered"
    assert isinstance(caught[0], httpx.ReadTimeout), (
        f"the call ended on {caught[0]!r} rather than a read timeout, so it proves "
        "nothing about the budget on a request that was accepted"
    )
    gc.collect()
    assert client_ref and client_ref[0]() is None, (
        "the timed-out client is still reachable after the probe returned -- the caught "
        "error carries the traceback that pins the frame that built it, so the next test "
        "to call gc.collect() inherits the aclose() task its finaliser schedules on "
        "whatever event loop is running then"
    )


def test_cloud_embedding_cleanup_is_noop():
    """No persistent local resource to release — must not raise and must
    not touch the client, matching CloudLLMProvider.cleanup()'s own no-op."""
    provider = CloudEmbeddingProvider(gemini_api_key="key", model="text-embedding-004")
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is not None




def test_local_embedding_model_name():
    provider = LocalEmbeddingProvider(
        ollama_host="http://localhost:11434", model="nomic-embed-text"
    )
    assert provider.model_name == "ollama/nomic-embed-text"


@pytest.mark.asyncio
async def test_local_embedding_embed_parses_response():
    """Exercises the real `_call_embed` parsing logic
    (`response["embedding"]`), not a mocked-away static method."""
    provider = LocalEmbeddingProvider(
        ollama_host="http://localhost:11434", model="nomic-embed-text"
    )

    fake_client = MagicMock()
    fake_client.embeddings.return_value = {"embedding": [0.4, 0.5, 0.6]}
    provider._client = fake_client

    result = await provider.embed("hello world")

    assert result == [0.4, 0.5, 0.6]
    fake_client.embeddings.assert_called_once_with(model="nomic-embed-text", prompt="hello world")


def test_local_embedding_cleanup_unloads_model_and_clears_client():
    """Mirrors LocalLLMProvider.cleanup(): unloads via keep_alive=0 and
    drops the client reference — the resource-hygiene measure that matters
    on the project's stated 8 GB unified-memory Local-mode target."""
    provider = LocalEmbeddingProvider(
        ollama_host="http://localhost:11434", model="nomic-embed-text"
    )
    fake_client = MagicMock()
    provider._client = fake_client

    provider.cleanup()

    fake_client.embeddings.assert_called_once_with(
        model="nomic-embed-text", prompt="", keep_alive=0
    )
    assert provider._client is None


def test_local_embedding_cleanup_swallows_errors():
    """A failed unload call (Ollama already stopped, connection refused)
    must not raise — cleanup is best-effort, same contract as
    LocalLLMProvider.cleanup()."""
    provider = LocalEmbeddingProvider(
        ollama_host="http://localhost:11434", model="nomic-embed-text"
    )
    fake_client = MagicMock()
    fake_client.embeddings.side_effect = RuntimeError("connection refused")
    provider._client = fake_client

    provider.cleanup()

    assert provider._client is None


def test_clear_cache_records_a_provider_cleanup_failure(caplog):
    """`LocalEmbeddingProvider.cleanup()` is an HTTP call to Ollama that unloads
    `nomic-embed-text`. A host that has gone away leaves the model resident, and
    the swallow left no record that the unload was even attempted."""
    import app.embeddings as embeddings_module

    provider = MagicMock()
    provider.cleanup.side_effect = OSError("Ollama is not reachable")
    embeddings_module._cached_provider = provider

    with caplog.at_level(logging.DEBUG, logger="app.embeddings"):
        clear_cache()

    failures = [r for r in caplog.records if r.name == "app.embeddings" and r.exc_info]
    assert len(failures) == 1
    assert embeddings_module._cached_provider is None


@pytest.mark.asyncio
async def test_a_stale_local_provider_that_refuses_to_release_is_recorded(caplog):
    """The re-probe branch's own swallow: the model has disappeared from Ollama,
    so the cached provider is dropped — and if its release fails, that is the
    same lost unload as above, on the path that runs without anyone asking."""
    import app.embeddings as embeddings_module

    stt, emb = _settings(ProviderMode.LOCAL)
    stale = MagicMock()
    stale.cleanup.side_effect = OSError("Ollama is not reachable")
    embeddings_module._cached_provider = stale
    embeddings_module._cached_key = ProviderMode.LOCAL

    with (
        patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=False)),
        caplog.at_level(logging.DEBUG, logger="app.embeddings"),
    ):
        provider, reason = await resolve_embedding_provider(stt, emb)

    assert provider is None
    assert reason == LOCAL_MISSING_MODEL_REASON
    failures = [r for r in caplog.records if r.name == "app.embeddings" and r.exc_info]
    assert len(failures) == 1

