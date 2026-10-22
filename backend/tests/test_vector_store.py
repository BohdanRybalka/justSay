"""Vector store — v3 DDL/migration, dim-guard trigger, model-switch wipe,
cascade-delete, relocate integrity, backfill resumability + clamping,
selftest (spec 003), automatic background indexing (spec 017 / ADR 010)."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import app.stt as stt_package
from app.config import settings
from app.core.types import ProviderMode
from app.stt import groq_whisper, local, local_whisper_cpp
from app.stt.base import STTProvider
from app.transcripts import history, relocation, schema, search, vector_store


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    history.bootstrap(tmp_path)
    yield
    with history._lock:
        history._close_conn_locked()


_GROQ_LABEL = "groq/whisper-large-v3-turbo"


class _FakeProvider:
    def __init__(
        self,
        model_name: str,
        vector: list[float] | None = None,
        fail: bool = False,
        is_local: bool = False,
    ):
        self.model_name = model_name
        self.is_local = is_local
        self.embedded: list[str] = []
        self._vector = vector or [0.1, 0.2, 0.3]
        self._fail = fail

    async def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        if self._fail:
            raise RuntimeError("provider outage")
        return self._vector


def _rowid(conn: sqlite3.Connection, entry_id: str) -> int:
    return conn.execute("SELECT rowid FROM entries WHERE id = ?", (entry_id,)).fetchone()[0]


def _embedded_labels() -> set[str | None]:
    with history._lock:
        conn = history._ensure_conn_locked()
        return {
            row[0]
            for row in conn.execute(
                "SELECT e.model_name FROM entries e JOIN entry_embeddings x ON x.entry_id = e.id"
            )
        }



def test_schema_version_is_current():
    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_v3_tables_and_trigger_exist_vec_entries_lazy():
    with history._lock:
        conn = history._ensure_conn_locked()
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names
    assert "entry_embeddings_dim_guard" in names
    assert "vec_entries" not in names


def test_migration_v1_to_current(tmp_path):
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(schema._DDL_V1)
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names


def test_migration_v2_to_current(tmp_path):
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(schema._DDL_V1)
        raw.executescript(schema._DDL_V2)
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names


def test_crash_before_the_version_pragma_retries(tmp_path):
    """A crash that ran the v3 DDL but never wrote user_version=3 must
    self-heal idempotently on the next boot (mirrors the existing v1->v2
    crash-safety test in test_history_sqlite.py)."""
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(schema._DDL_V1)
        raw.executescript(schema._DDL_V2)
        raw.executescript(vector_store._DDL_V3)
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_partial_migration_recovery_v3_tables_missing(tmp_path):
    """If a previous boot wrote user_version=3 but the v3 tables are
    missing (interrupted migration), _init_schema must recreate them via
    IF NOT EXISTS."""
    db_path = tmp_path / "history.db"
    with history._lock:
        history._close_conn_locked()
    db_path.unlink(missing_ok=True)
    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(schema._DDL_V1)
        raw.executescript(schema._DDL_V2)
        raw.execute("INSERT INTO entry_fts(entry_fts) VALUES('rebuild')")
        raw.execute("PRAGMA user_version = 3")
        raw.commit()
    finally:
        raw.close()

    history.bootstrap(tmp_path)

    with history._lock:
        conn = history._ensure_conn_locked()
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            ).fetchall()
        }
    assert "embeddings_meta" in names
    assert "entry_embeddings" in names
    assert "entry_embeddings_dim_guard" in names



def test_dim_guard_trigger_rejects_mismatched_dim():
    e = history.save_entry(text="hello", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO entry_embeddings(entry_id, model, dim, created_ts) "
                "VALUES (?, ?, ?, ?)",
                (e.id, "text-embedding-004", 5, 1),
            )


def test_dim_guard_trigger_allows_matching_dim():
    e = history.save_entry(text="hello", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        conn.execute(
            "INSERT INTO entry_embeddings(entry_id, model, dim, created_ts) VALUES (?, ?, ?, ?)",
            (e.id, "text-embedding-004", 3, 1),
        )
        stored = [
            tuple(row)
            for row in conn.execute(
                "SELECT model, dim FROM entry_embeddings WHERE entry_id = ?", (e.id,)
            )
        ]

    assert stored == [("text-embedding-004", 3)], (
        "a row whose dim matches the vec table passes the guard trigger and is "
        f"readable back at that dim: {stored}"
    )



def test_ensure_vec_table_idempotent_when_unchanged():
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        meta = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert tuple(meta) == ("cloud", "text-embedding-004", 3)


def test_ensure_vec_table_wipes_on_model_switch():
    e1 = history.save_entry(text="alpha searchable text", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e1.id, _rowid(conn, e1.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )
        count_before = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count_before == 1

    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "local", "nomic-embed-text", 5)
        count_after = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
        meta = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert count_after == 0
    assert tuple(meta) == ("local", "nomic-embed-text", 5)

    fts_hits = search.search_history("alpha", limit=5)
    assert len(fts_hits) == 1


@pytest.mark.asyncio
async def test_provider_switch_via_embed_background_wipes_old_embeddings():
    """Seeds embeddings under one model, flips the resolved provider,
    triggers a background embed, and asserts the old rows are gone and
    embeddings_meta reflects the new model. mode=fts still returns the
    pre-wipe entries."""
    e = history.save_entry(text="alpha searchable text", duration_ms=1, model_name=_GROQ_LABEL)

    provider_a = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])
    with (
        patch.object(settings.stt, "mode", ProviderMode.CLOUD),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(provider_a, None)),
        ),
    ):
        await vector_store.embed_entry_background(e.id, "alpha searchable text")

    with history._lock:
        conn = history._ensure_conn_locked()
        meta_before = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert tuple(meta_before) == ("cloud", "gemini/text-embedding-004", 3)

    provider_b = _FakeProvider(
        "ollama/nomic-embed-text", vector=[1.0, 2.0, 3.0, 4.0, 5.0], is_local=True
    )
    with (
        patch.object(settings.stt, "mode", ProviderMode.LOCAL),
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(provider_b, None)),
        ),
    ):
        await vector_store.embed_entry_background(e.id, "alpha searchable text")

    with history._lock:
        conn = history._ensure_conn_locked()
        count_after = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
        meta_after = conn.execute(
            "SELECT provider, model, dim FROM embeddings_meta WHERE id=1"
        ).fetchone()
    assert count_after == 1
    assert tuple(meta_after) == ("local", "ollama/nomic-embed-text", 5)

    fts_hits = search.search_history("alpha", limit=5)
    assert len(fts_hits) == 1



def test_delete_entry_cascades_to_entry_embeddings():
    e = history.save_entry(text="cascade me", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, _rowid(conn, e.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.delete_entry(e.id)

    with history._lock:
        conn = history._ensure_conn_locked()
        row = conn.execute("SELECT 1 FROM entry_embeddings WHERE entry_id = ?", (e.id,)).fetchone()
    assert row is None


def test_clear_all_cascades_to_entry_embeddings():
    e = history.save_entry(text="cascade me too", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, _rowid(conn, e.id), [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.clear_all()

    with history._lock:
        conn = history._ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count == 0


def test_delete_entry_removes_vec_entries_row_via_trigger():
    e = history.save_entry(text="vec cascade", duration_ms=1)
    with history._lock:
        conn = history._ensure_conn_locked()
        rowid = _rowid(conn, e.id)
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e.id, rowid, [1.0, 2.0, 3.0], "cloud", "text-embedding-004"
        )

    history.delete_entry(e.id)

    with history._lock:
        conn = history._ensure_conn_locked()
        row = conn.execute("SELECT 1 FROM vec_entries WHERE rowid = ?", (rowid,)).fetchone()
    assert row is None



def test_relocate_preserves_embeddings_and_semantic_search(tmp_path):
    e1 = history.save_entry(text="close match alpha", duration_ms=1)
    e2 = history.save_entry(text="totally different beta", duration_ms=1)

    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "cloud", "text-embedding-004", 3)
        vector_store.insert_embedding(
            conn, e1.id, _rowid(conn, e1.id), [1.0, 0.0, 0.0], "cloud", "text-embedding-004"
        )
        vector_store.insert_embedding(
            conn, e2.id, _rowid(conn, e2.id), [0.0, 1.0, 0.0], "cloud", "text-embedding-004"
        )

    new_dir = tmp_path / "new"
    res, _ = relocation.relocate(new_dir)
    assert res == relocation.RelocateOutcome.MOVED

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = vector_store.query_similar(conn, [1.0, 0.0, 0.0], 1)
    assert len(rows) == 1
    assert rows[0]["id"] == e1.id



@pytest.mark.asyncio
async def test_backfill_resumable_across_two_calls():
    for i in range(5):
        history.save_entry(text=f"resume entry {i}", duration_ms=1, model_name=_GROQ_LABEL)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        first = await vector_store.backfill_batch(2)
        assert first.processed == 2
        assert first.remaining == 3

        with history._lock:
            history._close_conn_locked()

        second = await vector_store.backfill_batch(2)
        assert second.processed == 2
        assert second.remaining == 1

        third = await vector_store.backfill_batch(2)
        assert third.processed == 1
        assert third.remaining == 0

    with history._lock:
        conn = history._ensure_conn_locked()
        total_embedded = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert total_embedded == 5


@pytest.mark.parametrize(
    ("meetings_mode", "embedded"),
    [(ProviderMode.LOCAL, {"dictation"}), (ProviderMode.CLOUD, {"dictation", "meeting"})],
)
async def test_a_meeting_kept_on_this_computer_never_reaches_a_cloud_embedder(
    monkeypatch, meetings_mode, embedded
):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    monkeypatch.setattr(settings.stt, "meetings_mode", meetings_mode)
    history.save_entry(text="dictated", duration_ms=1, model_name=_GROQ_LABEL)
    history.save_entry(
        text="said on the call", duration_ms=1, source="meeting", model_name=_GROQ_LABEL
    )
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(10)

    assert result.remaining == 0
    with history._lock:
        conn = history._ensure_conn_locked()
        sources = {
            row[0]
            for row in conn.execute(
                "SELECT e.source FROM entries e JOIN entry_embeddings x ON x.entry_id = e.id"
            )
        }
    assert sources == embedded


@pytest.mark.parametrize(
    ("files_mode", "embedded"),
    [(ProviderMode.LOCAL, {"dictation"}), (ProviderMode.CLOUD, {"dictation", "file"})],
)
async def test_a_file_kept_on_this_computer_never_reaches_a_cloud_embedder(
    monkeypatch, files_mode, embedded
):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    monkeypatch.setattr(settings.stt, "meetings_mode", ProviderMode.CLOUD)
    monkeypatch.setattr(settings.stt, "files_mode", files_mode)
    history.save_entry(text="dictated", duration_ms=1, model_name=_GROQ_LABEL)
    history.save_entry(
        text="from a recording",
        duration_ms=1,
        source="file",
        source_name="a.wav",
        model_name=_GROQ_LABEL,
    )
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(10)

    assert result.remaining == 0
    with history._lock:
        conn = history._ensure_conn_locked()
        sources = {
            row[0]
            for row in conn.execute(
                "SELECT e.source FROM entries e JOIN entry_embeddings x ON x.entry_id = e.id"
            )
        }
    assert sources == embedded


async def test_a_local_dictation_mode_embeds_a_file_on_this_computer(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    monkeypatch.setattr(settings.stt, "files_mode", ProviderMode.LOCAL)
    history.save_entry(text="from a recording", duration_ms=1, source="file", source_name="a.wav")
    fake = _FakeProvider("ollama/nomic-embed-text", vector=[1.0, 2.0, 3.0], is_local=True)

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(10)

    assert (result.processed, result.remaining) == (1, 0)


_LABELS_WRITTEN_HERE = (
    "whisper/large-v3",
    "whisper-mlx/large-v3",
    "whisper-cpp/large-v3-turbo",
    "whisper-cpp-vulkan/large-v3-turbo",
    "whisper-cpp-metal/large-v3-turbo",
    "ollama/llama3.2",
    None,
)
_LABELS_WRITTEN_IN_THE_CLOUD = {_GROQ_LABEL, "gemini/gemini-2.5-flash"}


async def test_an_entry_transcribed_here_never_reaches_a_cloud_embedder(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    for label in (*_LABELS_WRITTEN_HERE, *sorted(_LABELS_WRITTEN_IN_THE_CLOUD)):
        history.save_entry(text=f"said through {label}", duration_ms=1, model_name=label)
    fake = _FakeProvider("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(50)

    assert _embedded_labels() == _LABELS_WRITTEN_IN_THE_CLOUD
    assert sorted(fake.embedded) == sorted(
        f"said through {label}" for label in _LABELS_WRITTEN_IN_THE_CLOUD
    )
    assert result.remaining == 0, "entries a cloud embedder may not read are no backlog for it"


async def test_a_local_embedder_reads_every_entry_whoever_transcribed_it(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    for label in (*_LABELS_WRITTEN_HERE, *sorted(_LABELS_WRITTEN_IN_THE_CLOUD)):
        history.save_entry(text=f"said through {label}", duration_ms=1, model_name=label)
    fake = _FakeProvider("ollama/nomic-embed-text", vector=[1.0, 2.0, 3.0], is_local=True)

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(50)

    assert _embedded_labels() == {*_LABELS_WRITTEN_HERE, *_LABELS_WRITTEN_IN_THE_CLOUD}
    assert result.remaining == 0


async def test_a_cloud_embedder_resolved_after_a_switch_to_local_sends_nothing(monkeypatch):
    """The switch can flip between resolving the provider and embedding; the provider then
    says Cloud while the settings say Local, and Local wins, its vectors untouched."""
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    here = history.save_entry(text="dictated here", duration_ms=1, model_name="whisper/large-v3")
    history.save_entry(text="dictated in the cloud", duration_ms=1, model_name=_GROQ_LABEL)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "local", "ollama/nomic-embed-text", 3)
        rowid = _rowid(conn, here.id)
        vector_store.insert_embedding(
            conn, here.id, rowid, [1.0, 2.0, 3.0], "local", "ollama/nomic-embed-text"
        )
    fake = _FakeProvider("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(10)

    assert fake.embedded == []
    assert (result.processed, _embedded_labels()) == (0, {"whisper/large-v3"})


async def test_a_switch_to_local_stops_a_cloud_batch_already_running(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    for i in range(3):
        history.save_entry(text=f"cloud entry {i}", duration_ms=1, model_name=_GROQ_LABEL)

    class _SwitchedMidBatch(_FakeProvider):
        async def embed(self, text: str) -> list[float]:
            settings.stt.mode = ProviderMode.LOCAL
            return await super().embed(text)

    fake = _SwitchedMidBatch("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(10)

    assert (len(fake.embedded), result.processed) == (1, 1)


async def test_a_cloud_sweep_that_embeds_nothing_leaves_history_deletable(monkeypatch):
    """Without a key or a network the cloud embedder writes nothing, so the local vectors must
    stay whole until it does; a dropped vector table breaks every delete through its trigger."""
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    here = history.save_entry(text="dictated here", duration_ms=1, model_name="whisper/large-v3")
    cloud = history.save_entry(text="dictated in the cloud", duration_ms=1, model_name=_GROQ_LABEL)
    with history._lock:
        conn = history._ensure_conn_locked()
        vector_store.ensure_vec_table_locked(conn, "local", "ollama/nomic-embed-text", 3)
        rowid = _rowid(conn, here.id)
        vector_store.insert_embedding(
            conn, here.id, rowid, [1.0, 2.0, 3.0], "local", "ollama/nomic-embed-text"
        )
    failing = _FakeProvider("gemini/gemini-embedding-001", fail=True)

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(failing, None))
    ):
        await vector_store.backfill_batch(10)
    history.delete_entry(here.id)
    history.delete_entry(cloud.id)

    with history._lock:
        conn = history._ensure_conn_locked()
        left = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert (failing.embedded, left) == (["dictated in the cloud"], 0)


async def test_the_shipped_embedders_keep_entries_made_here_out_of_the_cloud(monkeypatch):
    """Real providers, only their network call stood in for: the locality each declares is
    what the rule reads, so a wrong declaration on either class fails here."""
    from app.embeddings import clear_cache as clear_embeddings_cache
    from app.embeddings.cloud import CloudEmbeddingProvider
    from app.embeddings.local import LocalEmbeddingProvider

    history.save_entry(text="dictated here", duration_ms=1, model_name="whisper/large-v3")
    history.save_entry(text="dictated in the cloud", duration_ms=1, model_name=_GROQ_LABEL)
    sent: dict[str, list[str]] = {"cloud": [], "local": []}

    def recorder(where: str):
        def call_embed(_client, _model: str, text: str) -> list[float]:
            sent[where].append(text)
            return [1.0, 2.0, 3.0]

        return staticmethod(call_embed)

    monkeypatch.setattr(CloudEmbeddingProvider, "_get_client", lambda self: object())
    monkeypatch.setattr(CloudEmbeddingProvider, "_call_embed", recorder("cloud"))
    monkeypatch.setattr(LocalEmbeddingProvider, "_get_client", lambda self: object())
    monkeypatch.setattr(LocalEmbeddingProvider, "_call_embed", recorder("local"))
    monkeypatch.setattr("app.embeddings.local.is_model_available", AsyncMock(return_value=True))
    clear_embeddings_cache()
    try:
        monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
        await vector_store.backfill_batch(10)
        with history._lock:
            history._ensure_conn_locked().execute("DELETE FROM entry_embeddings")
        monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
        await vector_store.backfill_batch(10)
    finally:
        clear_embeddings_cache()

    assert sent["cloud"] == ["dictated in the cloud"]
    assert sorted(sent["local"]) == ["dictated here", "dictated in the cloud"]


def _speech_engine_classes() -> set[type[STTProvider]]:
    for module in pkgutil.iter_modules(stt_package.__path__):
        importlib.import_module(f"{stt_package.__name__}.{module.name}")
    found: set[type[STTProvider]] = set()
    pending = list(STTProvider.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("app.") and not inspect.isabstract(cls):
            found.add(cls)
    return found


async def test_every_speech_engine_label_reaches_a_cloud_embedder_only_if_it_ran_elsewhere(
    monkeypatch,
):
    engines = _speech_engine_classes()
    assert {
        groq_whisper.GroqWhisperSTTProvider,
        local.LocalSTTProvider,
        local_whisper_cpp.WhisperCppServerSTTProvider,
    } <= engines
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    labels = {engine: engine(settings.stt).model_name for engine in engines}
    for label in labels.values():
        history.save_entry(text=f"said through {label}", duration_ms=1, model_name=label)
    fake = _FakeProvider("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        await vector_store.backfill_batch(50)

    assert _embedded_labels() == {
        label for engine, label in labels.items() if not engine.is_local
    }, f"a speech engine's stored label disagrees with where it runs: {labels}"


async def test_a_dictation_made_here_is_not_embedded_in_the_cloud_after_the_switch(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.CLOUD)
    e = history.save_entry(text="dictated here", duration_ms=1, model_name="whisper/large-v3")
    fake = _FakeProvider("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        await vector_store.embed_entry_background(e.id, "dictated here")

    assert fake.embedded == []
    assert _embedded_labels() == set()


async def test_a_dictation_made_here_is_embedded_at_once_by_a_local_embedder(monkeypatch):
    monkeypatch.setattr(settings.stt, "mode", ProviderMode.LOCAL)
    e = history.save_entry(text="dictated here", duration_ms=1, model_name="whisper/large-v3")
    fake = _FakeProvider("ollama/nomic-embed-text", vector=[1.0, 2.0, 3.0], is_local=True)

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        await vector_store.embed_entry_background(e.id, "dictated here")

    assert _embedded_labels() == {"whisper/large-v3"}


async def test_an_entry_deleted_while_its_embedding_is_made_leaves_no_row(caplog):
    e = history.save_entry(text="delete me midway", duration_ms=1, model_name=_GROQ_LABEL)

    class _DeletesItsEntry(_FakeProvider):
        async def embed(self, text: str) -> list[float]:
            history.delete_entry(e.id)
            return await super().embed(text)

    fake = _DeletesItsEntry("gemini/gemini-embedding-001", vector=[1.0, 2.0, 3.0])
    with (
        caplog.at_level(logging.WARNING, logger=vector_store.log.name),
        patch(
            "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
        ),
    ):
        await vector_store.embed_entry_background(e.id, "delete me midway")

    with history._lock:
        conn = history._ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert (fake.embedded, count) == (["delete me midway"], 0)
    assert not caplog.records, [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_backfill_batch_size_clamped_to_200():
    for i in range(205):
        history.save_entry(text=f"entry-{i}", duration_ms=1, model_name=_GROQ_LABEL)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
    ):
        result = await vector_store.backfill_batch(99999)

    assert result.processed == 200
    assert result.remaining == 5
    assert vector_store.BACKFILL_BATCH_MAX == 200


@pytest.mark.asyncio
async def test_backfill_noop_when_disabled():
    history.save_entry(text="x", duration_ms=1, model_name=_GROQ_LABEL)
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(None, "disabled"))
    ):
        result = await vector_store.backfill_batch(50)
    assert result.processed == 0
    assert result.remaining == 1



@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_drains_backlog():
    """Seeds more not-yet-embedded entries than one batch (forcing several
    ``backfill_batch`` loop iterations) with a working mock provider, and
    asserts the backlog reaches 0 after the call."""
    entry_count = vector_store._INDEXER_BATCH_SIZE * 2 + 5
    for i in range(entry_count):
        history.save_entry(text=f"drain entry {i}", duration_ms=1, model_name=_GROQ_LABEL)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    with (
        patch(
            "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
        ),
        patch.object(vector_store, "_INDEXER_PACING_SECONDS", 0),
    ):
        await vector_store.run_background_indexer()

    with history._lock:
        conn = history._ensure_conn_locked()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE id NOT IN (SELECT entry_id FROM entry_embeddings)"
        ).fetchone()[0]
    assert remaining == 0


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_stops_after_one_stalled_batch():
    """A never-succeeding mock provider must not spin forever — the
    function returns after exactly one stalled batch (processed == 0 while
    remaining > 0), not an infinite loop."""
    for i in range(3):
        history.save_entry(text=f"stalled entry {i}", duration_ms=1, model_name=_GROQ_LABEL)
    failing = _FakeProvider("gemini/text-embedding-004", fail=True)

    with (
        patch(
            "app.embeddings.resolve_embedding_provider",
            new=AsyncMock(return_value=(failing, None)),
        ),
        patch.object(vector_store, "backfill_batch", wraps=vector_store.backfill_batch) as spy,
    ):
        await vector_store.run_background_indexer()

    assert spy.await_count == 1

    with history._lock:
        conn = history._ensure_conn_locked()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE id NOT IN (SELECT entry_id FROM entry_embeddings)"
        ).fetchone()[0]
    assert remaining == 3


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_noop_when_vec_unavailable():
    history.save_entry(text="x", duration_ms=1)
    with (
        patch.object(history, "_vec_available", False),
        patch.object(vector_store, "backfill_batch") as batch_mock,
    ):
        await vector_store.run_background_indexer()
    batch_mock.assert_not_called()


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_serializes_concurrent_calls():
    """Two concurrent calls (simulating two dictations completing back-to-
    back) must never execute their backfill_batch loop bodies concurrently
    — serialized through _indexer_lock, mirroring spec 013's
    _local_reprobe_lock concurrency-safety bar."""
    for i in range(4):
        history.save_entry(text=f"concurrent entry {i}", duration_ms=1, model_name=_GROQ_LABEL)
    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])

    active = 0
    max_active = 0

    real_backfill_batch = vector_store.backfill_batch

    async def tracked_backfill_batch(batch_size):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.05)
            return await real_backfill_batch(batch_size)
        finally:
            active -= 1

    with (
        patch(
            "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
        ),
        patch.object(vector_store, "backfill_batch", side_effect=tracked_backfill_batch),
        patch.object(vector_store, "_INDEXER_PACING_SECONDS", 0),
    ):
        await asyncio.gather(
            vector_store.run_background_indexer(),
            vector_store.run_background_indexer(),
        )

    assert max_active == 1


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_swallows_backfill_exception(caplog):
    """Same "must never raise" contract as embed_entry_background: an
    exception raised inside backfill_batch (e.g. sqlite3.OperationalError
    from cross-thread history._lock contention — see this spec's Risks
    section) must be caught and logged at warning, never propagated through
    the BackgroundTasks/asyncio.create_task entrypoint."""
    history.save_entry(text="x", duration_ms=1)
    with (
        patch.object(vector_store, "backfill_batch", side_effect=RuntimeError("boom")),
        caplog.at_level(logging.WARNING, logger="app.transcripts.vector_store"),
    ):
        await vector_store.run_background_indexer()

    assert any(
        "Background indexer sweep failed" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.background_indexer
@pytest.mark.asyncio
async def test_run_background_indexer_cancelled_mid_sweep_loses_no_committed_work():
    """Spec 036: shutdown now cancels this sweep mid-flight, so the cancel
    must be safe. Every entry embedded before the cancel keeps its
    `entry_embeddings` row (history's connection is autocommit and
    `insert_embedding` contains no `await`, so a cancellation cannot land
    part-way through one row's write), and a fresh run finishes the rest.

    Also pins the claim that `CancelledError` is NOT swallowed by
    `run_background_indexer`'s `except Exception` -- it derives from
    `BaseException`, so it propagates and the task ends up genuinely
    cancelled rather than logged as a spurious failure.
    """
    for i in range(12):
        history.save_entry(text=f"cancel entry {i}", duration_ms=1, model_name=_GROQ_LABEL)

    reached_sixth = asyncio.Event()

    class _BlockingProvider:
        model_name = "gemini/text-embedding-004"
        is_local = False

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, text: str) -> list[float]:
            self.calls += 1
            if self.calls > 5:
                reached_sixth.set()
                await asyncio.sleep(30)
            return [1.0, 2.0, 3.0]

    with patch(
        "app.embeddings.resolve_embedding_provider",
        new=AsyncMock(return_value=(_BlockingProvider(), None)),
    ):
        # background-task-ok: test fixture, awaited below
        sweep = asyncio.create_task(vector_store.run_background_indexer())
        await asyncio.wait_for(reached_sixth.wait(), timeout=5)
        sweep.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sweep

    with history._lock:
        conn = history._ensure_conn_locked()
        embedded = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert embedded == 5, "committed rows were lost by the mid-sweep cancel"

    fake = _FakeProvider("gemini/text-embedding-004", vector=[1.0, 2.0, 3.0])
    with (
        patch(
            "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
        ),
        patch.object(vector_store, "_INDEXER_PACING_SECONDS", 0),
    ):
        await vector_store.run_background_indexer()

    with history._lock:
        conn = history._ensure_conn_locked()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE id NOT IN (SELECT entry_id FROM entry_embeddings)"
        ).fetchone()[0]
    assert remaining == 0



@pytest.mark.asyncio
async def test_embed_entry_background_never_raises_on_provider_failure():
    e = history.save_entry(text="will fail", duration_ms=1, model_name=_GROQ_LABEL)
    failing = _FakeProvider("gemini/text-embedding-004", fail=True)
    with patch(
        "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(failing, None))
    ):
        await vector_store.embed_entry_background(e.id, "will fail")

    with history._lock:
        conn = history._ensure_conn_locked()
        count = conn.execute("SELECT COUNT(*) FROM entry_embeddings").fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_embed_entry_background_skips_deleted_entry(caplog):
    """If the entry was deleted before the background task ran, nothing is embedded
    and nothing raises: the entry check finds no row before the provider is called."""
    fake = _FakeProvider("gemini/text-embedding-004")
    with (
        caplog.at_level(logging.WARNING, logger=vector_store.log.name),
        patch(
            "app.embeddings.resolve_embedding_provider", new=AsyncMock(return_value=(fake, None))
        ),
    ):
        await vector_store.embed_entry_background("does-not-exist", "text")

    with history._lock:
        conn = history._ensure_conn_locked()
        rows = conn.execute(
            "SELECT entry_id FROM entry_embeddings WHERE entry_id = ?", ("does-not-exist",)
        ).fetchall()

    assert rows == [], (
        "an entry deleted before its background embed ran must leave no row behind, "
        f"rather than one pointing at an entry_id nothing resolves: {rows}"
    )
    assert fake.embedded == []
    assert not caplog.records, (
        "the missing row must be skipped by the entry check, not swallowed by the "
        "failure handler -- both leave the table empty, so only the absence of a "
        f"warning tells them apart: {[record.getMessage() for record in caplog.records]}"
    )



def test_selftest_ok():
    ok, msg = vector_store.selftest()
    assert ok is True, msg
    assert msg == "ok"


def test_selftest_never_raises_on_broken_extension(monkeypatch):
    def boom(_conn):
        raise RuntimeError("extension load disabled on this platform")

    monkeypatch.setattr(vector_store.sqlite_vec, "load", boom)
    ok, msg = vector_store.selftest()
    assert ok is False
    assert "extension load disabled" in msg


def test_selftest_fails_a_bundle_whose_sqlite_predates_row_values(monkeypatch):
    """History paging seeks with ``(ts, id) < (?, ?)``, which SQLite gained in
    3.15.0; below that every history read is a syntax error rather than a wrong
    answer. Which library a PyInstaller build embeds only a packaged build can
    answer, so the floor is asserted where ``release.yml`` already runs the frozen
    binary on both platforms -- not on a startup or request path, where a version
    failure would take dictation and settings down with it.
    """
    monkeypatch.setattr(vector_store.sqlite3, "sqlite_version_info", (3, 14, 0))
    monkeypatch.setattr(vector_store.sqlite3, "sqlite_version", "3.14.0")

    ok, msg = vector_store.selftest()

    assert ok is False
    assert "3.14.0" in msg
    assert "3.15" in msg


def test_selftest_accepts_this_environments_sqlite():
    assert sqlite3.sqlite_version_info >= history.ROW_VALUE_MIN_SQLITE_VERSION
    assert vector_store.selftest() == (True, "ok")


def test_selftest_reports_the_extension_outcome_even_on_an_old_sqlite(monkeypatch):
    """An old bundled library is exactly the build whose sqlite-vec status is worth
    knowing, so the version check must not short-circuit the extension probe: one
    release run has to answer both questions rather than hide the second behind the
    first."""

    def boom(_conn):
        raise RuntimeError("extension load disabled on this platform")

    monkeypatch.setattr(vector_store.sqlite3, "sqlite_version_info", (3, 14, 0))
    monkeypatch.setattr(vector_store.sqlite3, "sqlite_version", "3.14.0")
    monkeypatch.setattr(vector_store.sqlite_vec, "load", boom)

    ok, msg = vector_store.selftest()

    assert ok is False
    assert "3.14.0" in msg
    assert "extension load disabled" in msg


def test_selftest_fails_a_bundle_whose_planner_walks_the_cursor_read(monkeypatch):
    """The version floor names the release that *introduced* row values, not one at
    which the planner is known to seek ``entries_ts_id_idx`` for
    ``(ts, id) < (?, ?)``. A library that parses the predicate and then walks the
    index from the top gives back the whole property this paging buys, silently and
    without an error, so the gate takes a real plan from the loaded library."""
    monkeypatch.setattr(
        vector_store.history, "cursor_seek_plan_failure", lambda: "SCAN entries"
    )

    ok, msg = vector_store.selftest()

    assert ok is False
    assert "SCAN entries" in msg


def test_selftest_never_raises_when_the_seek_probe_does(monkeypatch):
    def boom():
        raise RuntimeError("no such table: entries")

    monkeypatch.setattr(vector_store.history, "cursor_seek_plan_failure", boom)

    ok, msg = vector_store.selftest()

    assert ok is False
    assert "no such table: entries" in msg


def test_selftest_never_raises_when_a_check_forgets_to_wrap_itself(monkeypatch):
    """The three checks each guard themselves today, so the loop's own guard is
    what keeps ``selftest``'s "Never raises" true for the fourth one."""

    def boom():
        raise RuntimeError("sqlite3 module is unusable")

    monkeypatch.setattr(vector_store, "_row_value_version_failure", boom)

    ok, msg = vector_store.selftest()

    assert ok is False
    assert "sqlite3 module is unusable" in msg


def test_the_cursor_seek_probe_passes_on_this_environments_sqlite():
    assert history.cursor_seek_plan_failure() is None
