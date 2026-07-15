"""Embedding provider selection — privacy eligibility matrix (spec 003).

Single most important AC of this spec: eligibility must be derived STRICTLY
from (stt.mode, llm.mode) with no cloud bypass hiding inside the local
branch. Mocks the factory's internal constructors directly (the same
pattern test_words.py uses for get_llm_provider) so a direct cloud-SDK
bypass cannot pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.embeddings import LOCAL_MISSING_MODEL_REASON, clear_cache, resolve_embedding_provider
from app.embeddings.cloud import CloudEmbeddingProvider
from app.embeddings.config import EmbeddingSettings
from app.embeddings.local import LocalEmbeddingProvider
from app.llm.config import LLMSettings
from app.stt.config import STTSettings


@pytest.fixture(autouse=True)
def _clear_embeddings_cache():
    clear_cache()
    yield
    clear_cache()


def _settings(stt_mode: ProviderMode, llm_mode: ProviderMode):
    stt = STTSettings(mode=stt_mode, gemini_api_key="key")
    llm = LLMSettings(mode=llm_mode)
    emb = EmbeddingSettings()
    return stt, llm, emb


MATRIX = [
    (ProviderMode.CLOUD, ProviderMode.CLOUD),
    (ProviderMode.CLOUD, ProviderMode.LOCAL),
    (ProviderMode.LOCAL, ProviderMode.CLOUD),
    (ProviderMode.LOCAL, ProviderMode.LOCAL),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stt_mode,llm_mode", MATRIX)
async def test_eligibility_matrix(stt_mode, llm_mode):
    """resolve_embedding_provider returns a CloudEmbeddingProvider ONLY for
    (cloud, cloud); a LocalEmbeddingProvider ONLY for (local, local) with
    Ollama reporting nomic-embed-text installed; None for every other pair,
    including both mixed pairings.
    """
    clear_cache()
    stt, llm, emb = _settings(stt_mode, llm_mode)

    fake_cloud = MagicMock(name="CloudEmbeddingProvider-instance")
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.cloud.CloudEmbeddingProvider", return_value=fake_cloud) as cloud_ctor,
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local) as local_ctor,
        patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)) as avail,
    ):
        provider, reason = await resolve_embedding_provider(stt, llm, emb)

    if (stt_mode, llm_mode) == (ProviderMode.CLOUD, ProviderMode.CLOUD):
        assert provider is fake_cloud
        cloud_ctor.assert_called_once()
        local_ctor.assert_not_called()
        avail.assert_not_called()
        assert reason is None
    elif (stt_mode, llm_mode) == (ProviderMode.LOCAL, ProviderMode.LOCAL):
        assert provider is fake_local
        avail.assert_called_once()
        local_ctor.assert_called_once()
        cloud_ctor.assert_not_called()
        assert reason is None
    else:
        assert provider is None
        cloud_ctor.assert_not_called()
        local_ctor.assert_not_called()
        assert reason is not None


@pytest.mark.asyncio
async def test_local_mode_disabled_when_model_not_pulled():
    """(local, local) with Ollama NOT reporting nomic-embed-text -> None +
    a specific actionable reason, and LocalEmbeddingProvider is never
    constructed (no half-built client left hanging around)."""
    stt, llm, emb = _settings(ProviderMode.LOCAL, ProviderMode.LOCAL)

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider") as local_ctor,
        patch("app.embeddings.local.is_model_available", new=AsyncMock(return_value=False)),
    ):
        provider, reason = await resolve_embedding_provider(stt, llm, emb)

    assert provider is None
    local_ctor.assert_not_called()
    assert reason is not None
    assert "nomic-embed-text" in reason


@pytest.mark.asyncio
async def test_mixed_mode_never_falls_back_to_cloud():
    """(cloud stt, local llm) and (local stt, cloud llm) must both disable
    the feature outright — no 'pick a side' fallback to cloud, which would
    be the actual privacy leak this AC exists to prevent."""
    for stt_mode, llm_mode in [
        (ProviderMode.CLOUD, ProviderMode.LOCAL),
        (ProviderMode.LOCAL, ProviderMode.CLOUD),
    ]:
        clear_cache()
        stt, llm, emb = _settings(stt_mode, llm_mode)
        with (
            patch("app.embeddings.cloud.CloudEmbeddingProvider") as cloud_ctor,
            patch("app.embeddings.local.LocalEmbeddingProvider") as local_ctor,
        ):
            provider, reason = await resolve_embedding_provider(stt, llm, emb)
        assert provider is None
        cloud_ctor.assert_not_called()
        local_ctor.assert_not_called()
        assert reason is not None
        assert "Cloud/Local mode" in reason


@pytest.mark.asyncio
async def test_factory_caches_provider_by_mode_pair():
    stt, llm, emb = _settings(ProviderMode.CLOUD, ProviderMode.CLOUD)
    with patch("app.embeddings.cloud.CloudEmbeddingProvider", return_value=MagicMock()):
        p1, _ = await resolve_embedding_provider(stt, llm, emb)
        p2, _ = await resolve_embedding_provider(stt, llm, emb)
    assert p1 is p2


@pytest.mark.asyncio
async def test_clear_cache_forces_reresolve():
    stt, llm, emb = _settings(ProviderMode.CLOUD, ProviderMode.CLOUD)
    with patch(
        "app.embeddings.cloud.CloudEmbeddingProvider", side_effect=lambda **_: MagicMock()
    ) as ctor:
        await resolve_embedding_provider(stt, llm, emb)
        clear_cache()
        await resolve_embedding_provider(stt, llm, emb)
    assert ctor.call_count == 2


@pytest.mark.asyncio
async def test_stale_unavailable_cache_reprobes_and_flips_available():
    """A cached (LOCAL, LOCAL) negative result caused by a missing model
    must not be served verbatim forever — it must re-probe Ollama on every
    call until the model appears, then cache the resulting provider
    normally, without an intervening clear_cache()."""
    stt, llm, emb = _settings(ProviderMode.LOCAL, ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available",
            new=AsyncMock(side_effect=[False, True]),
        ) as avail,
    ):
        provider1, reason1 = await resolve_embedding_provider(stt, llm, emb)
        provider2, reason2 = await resolve_embedding_provider(stt, llm, emb)

    assert (provider1, reason1) == (None, LOCAL_MISSING_MODEL_REASON)
    assert provider2 is fake_local
    assert reason2 is None
    assert avail.call_count == 2


@pytest.mark.asyncio
async def test_available_cache_reprobes_and_flips_to_unavailable():
    """A cached positive (LOCAL, LOCAL) result must not be served verbatim
    forever either — it re-probes on every call same as the negative branch,
    and flips to LOCAL_MISSING_MODEL_REASON (cleaning up the now-stale
    provider) when the model disappears (e.g. `ollama rm nomic-embed-text`).
    This supersedes the old "does not reprobe" assertion on purpose — that
    was exactly the gap this item closes, not a regression."""
    stt, llm, emb = _settings(ProviderMode.LOCAL, ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch("app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local),
        patch(
            "app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)
        ) as avail,
    ):
        provider1, reason1 = await resolve_embedding_provider(stt, llm, emb)
        assert avail.call_count == 1

        avail.return_value = False
        provider2, reason2 = await resolve_embedding_provider(stt, llm, emb)

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
    stt, llm, emb = _settings(ProviderMode.LOCAL, ProviderMode.LOCAL)
    fake_local = MagicMock(name="LocalEmbeddingProvider-instance")

    with (
        patch(
            "app.embeddings.local.LocalEmbeddingProvider", return_value=fake_local
        ) as local_ctor,
        patch(
            "app.embeddings.local.is_model_available", new=AsyncMock(return_value=True)
        ) as avail,
    ):
        provider1, _ = await resolve_embedding_provider(stt, llm, emb)
        provider2, _ = await resolve_embedding_provider(stt, llm, emb)

    assert provider1 is provider2 is fake_local
    local_ctor.assert_called_once()
    assert avail.call_count == 2
    fake_local.cleanup.assert_not_called()


# --- "Disabled means disabled" — closes the silent-fallback failure mode --

@pytest.mark.asyncio
async def test_embed_entry_background_noop_when_disabled():
    """When resolve_embedding_provider returns None, embed_entry_background
    must be a no-op that never imports/instantiates CloudEmbeddingProvider —
    closes the 'silent fallback to cloud' failure mode explicitly, not just
    by absence of a code path."""
    from app.core import history, vector_store

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
    from app.core import history, vector_store

    with (
        patch.object(history, "_vec_available", False),
        patch("app.embeddings.resolve_embedding_provider") as resolve_mock,
    ):
        await vector_store.embed_entry_background("entry-id", "some text")

    resolve_mock.assert_not_called()


# --- CloudEmbeddingProvider — direct unit tests -----------------------------
# Every test above replaces this class with a mock; these exercise the real
# SDK-response-parsing code (`response.embeddings[0].values`) and the
# missing-API-key `RuntimeError` path in `_get_client`, mirroring the
# pattern test_llm.py already uses for CloudLLMProvider.


def test_cloud_embedding_model_name():
    provider = CloudEmbeddingProvider(gemini_api_key="key", model="text-embedding-004")
    assert provider.model_name == "gemini/text-embedding-004"


def test_cloud_embedding_requires_api_key():
    provider = CloudEmbeddingProvider(gemini_api_key="", model="text-embedding-004")
    with pytest.raises(RuntimeError, match="missing"):
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
    provider._client = fake_client  # skip _get_client

    result = await provider.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    fake_client.models.embed_content.assert_called_once_with(
        model="text-embedding-004", contents="hello world"
    )


def test_cloud_embedding_cleanup_is_noop():
    """No persistent local resource to release — must not raise and must
    not touch the client, matching CloudLLMProvider.cleanup()'s own no-op."""
    provider = CloudEmbeddingProvider(gemini_api_key="key", model="text-embedding-004")
    provider._client = MagicMock()
    provider.cleanup()
    assert provider._client is not None


# --- LocalEmbeddingProvider — direct unit tests -----------------------------


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
    provider._client = fake_client  # skip _get_client

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

    provider.cleanup()  # must not raise

    assert provider._client is None
