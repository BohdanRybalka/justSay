"""Where each history entry came from: the v6 columns, the v5 -> v6 step, the writers' fields.

The v5 store below is written out rather than built from ``schema``: the point of the
fixture is a shape this build no longer declares.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.transcripts import history, relocation, schema, vector_store

_V5_ENTRIES_AS_SHIPPED = """
CREATE TABLE entries (
  id TEXT PRIMARY KEY NOT NULL CHECK (typeof(id) = 'text'),
  ts INTEGER NOT NULL CHECK (typeof(ts) = 'integer' AND ts BETWEEN 0 AND 253402300799000),
  language TEXT NOT NULL CHECK (typeof(language) = 'text'),
  raw_text TEXT NOT NULL CHECK (typeof(raw_text) = 'text'),
  cleaned_text TEXT NOT NULL CHECK (typeof(cleaned_text) = 'text'),
  duration_ms INTEGER NOT NULL CHECK (typeof(duration_ms) = 'integer' AND duration_ms >= 0),
  audio_duration_seconds REAL CHECK (
    audio_duration_seconds IS NULL OR typeof(audio_duration_seconds) IN ('integer', 'real')
  ),
  word_count INTEGER CHECK (word_count IS NULL OR typeof(word_count) = 'integer'),
  model_name TEXT CHECK (model_name IS NULL OR typeof(model_name) = 'text'),
  tokens_used INTEGER CHECK (tokens_used IS NULL OR typeof(tokens_used) = 'integer')
);
CREATE INDEX entries_ts_id_idx ON entries(ts DESC, id DESC);
CREATE VIRTUAL TABLE entry_fts USING fts5(
  cleaned_text,
  content='entries', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entry_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
CREATE TRIGGER entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text)
  VALUES('delete', old.rowid, old.cleaned_text);
END;
CREATE TRIGGER entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entry_fts(entry_fts, rowid, cleaned_text)
  VALUES('delete', old.rowid, old.cleaned_text);
  INSERT INTO entry_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
"""

_V5_ROWS = (
    ("first", "a faithful transcript about tomatoes"),
    ("second", "a longer transcript about cucumbers"),
    ("third", "another transcript about peppers"),
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_page_total_cache", None)
    yield
    with history._lock:
        history._close_conn_locked()


def _seed_a_v5_store(db_path) -> None:
    """Three rows as v5 stored them, one embedding each addressed by ``rowid``, a gap in
    the rowids so a renumbering cannot pass by accident."""
    import sqlite_vec

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(_V5_ENTRIES_AS_SHIPPED)
        raw.enable_load_extension(True)
        sqlite_vec.load(raw)
        raw.enable_load_extension(False)
        raw.executescript(vector_store._DDL_V3)
        raw.execute("CREATE VIRTUAL TABLE vec_entries USING vec0(embedding float[4])")
        raw.execute("INSERT INTO embeddings_meta(id, provider, model, dim) VALUES (1,'p','m',4)")
        for index, (entry_id, text) in enumerate(_V5_ROWS):
            rowid = 10 + index * 7
            raw.execute(
                "INSERT INTO entries(rowid, id, ts, language, raw_text, cleaned_text, "
                "duration_ms, audio_duration_seconds, word_count, model_name, tokens_used) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (rowid, entry_id, 1700000000000 + index, "uk", text, text, 10, 1.0, 5, "m", None),
            )
            raw.execute(
                "INSERT INTO entry_embeddings(entry_id, model, dim, created_ts) VALUES (?,'m',4,1)",
                (entry_id,),
            )
            raw.execute(
                "INSERT INTO vec_entries(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4])),
            )
        raw.execute("PRAGMA user_version = 5")
        raw.commit()
    finally:
        raw.close()


def _snapshot(db_path) -> dict:
    import sqlite_vec

    raw = sqlite3.connect(db_path)
    try:
        raw.enable_load_extension(True)
        sqlite_vec.load(raw)
        return {
            "rowids": dict(raw.execute("SELECT id, rowid FROM entries").fetchall()),
            "embeddings": raw.execute(
                "SELECT entry_id FROM entry_embeddings ORDER BY entry_id"
            ).fetchall(),
            "vectors": sorted(r[0] for r in raw.execute("SELECT rowid FROM vec_entries")),
            "version": raw.execute("PRAGMA user_version").fetchone()[0],
            "columns": [r[1] for r in raw.execute("PRAGMA table_info(entries)")],
        }
    finally:
        raw.close()


def _fts_hits(conn: sqlite3.Connection, word: str) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT e.id FROM entry_fts JOIN entries e ON e.rowid = entry_fts.rowid "
            "WHERE entry_fts MATCH ? ORDER BY e.id",
            (word,),
        )
    ]


def test_a_v5_store_keeps_every_row_its_rowid_and_its_embeddings(tmp_path):
    db = tmp_path / "history.db"
    _seed_a_v5_store(db)
    before = _snapshot(db)

    history.bootstrap(tmp_path)
    with history._lock:
        history._close_conn_locked()
    after = _snapshot(db)

    assert after["rowids"] == before["rowids"]
    assert after["embeddings"] == before["embeddings"]
    assert after["vectors"] == before["vectors"] == sorted(before["rowids"].values())
    assert after["version"] == schema.SCHEMA_VERSION == 6
    assert after["columns"] == list(schema.ENTRY_COLUMNS)
    entries = history.get_page(limit=10).entries
    assert {e.id: e.text for e in entries} == dict(_V5_ROWS)
    assert {(e.source, e.source_name, e.starred) for e in entries} == {("dictation", None, False)}
    with history._lock:
        assert _fts_hits(history._ensure_conn_locked(), "cucumbers") == ["second"]


def test_a_v5_store_gains_its_columns_without_a_table_rebuild(tmp_path, monkeypatch):
    _seed_a_v5_store(tmp_path / "history.db")
    rebuilds = []
    monkeypatch.setattr(
        schema, "_rebuild_entries_locked", lambda conn: rebuilds.append(conn) or False
    )

    history.bootstrap(tmp_path)

    assert rebuilds == []
    assert history.get_page(limit=10).entries[0].source == "dictation"


def test_a_failed_column_add_leaves_the_whole_store_at_v5(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    _seed_a_v5_store(db)
    before = _snapshot(db)
    monkeypatch.setattr(
        schema,
        "_V6_ADDED_COLUMNS",
        (schema._V6_ADDED_COLUMNS[0], ("starred", "INTEGER NOT NULL")),
    )

    history.bootstrap(tmp_path)
    with history._lock:
        in_transaction = history._ensure_conn_locked().in_transaction
        history._close_conn_locked()

    assert not in_transaction
    assert _snapshot(db) == before, "the column added before the failure is rolled back too"


def test_a_v5_store_is_copied_before_the_change_and_the_copy_goes_on_the_next_open(tmp_path):
    db = tmp_path / "history.db"
    _seed_a_v5_store(db)
    backup = tmp_path / "history.v5.bak"

    history.bootstrap(tmp_path)

    assert schema.migration_backup_path(db) == backup
    copy = _snapshot(backup)
    assert copy["version"] == 5
    assert copy["rowids"] == _snapshot(db)["rowids"]
    assert "source" not in copy["columns"]

    history.bootstrap(tmp_path)

    assert not backup.exists()


def test_a_fresh_store_leaves_no_backup(tmp_path):
    history.bootstrap(tmp_path)
    history.bootstrap(tmp_path)

    assert list(tmp_path.glob("*.bak")) == []


def test_relocating_the_store_takes_the_backup_away_from_the_old_folder(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    _seed_a_v5_store(old_dir / "history.db")
    history.bootstrap(old_dir)
    assert (old_dir / "history.v5.bak").exists()

    outcome, _ = relocation.relocate(tmp_path / "new")

    assert outcome is relocation.RelocateOutcome.MOVED
    assert list(old_dir.iterdir()) == []


@pytest.mark.parametrize("migrated", [False, True], ids=["fresh", "from-v5"])
def test_starring_leaves_the_search_index_alone(tmp_path, migrated):
    if migrated:
        _seed_a_v5_store(tmp_path / "history.db")
    history.bootstrap(tmp_path)
    entry = history.save_entry("tomatoes and cucumbers", duration_ms=1)

    with history._lock:
        conn = history._ensure_conn_locked()
        before = conn.total_changes
        conn.execute("UPDATE entries SET starred = 1 WHERE id = ?", (entry.id,))
        starring_changes = conn.total_changes - before
        conn.execute(
            "UPDATE entries SET cleaned_text = 'onions only' WHERE id = ?", (entry.id,)
        )
        onion_hits = _fts_hits(conn, "onions")
        tomato_hits = _fts_hits(conn, "tomatoes")

    assert starring_changes == 1, "the star wrote the row and nothing else"
    assert onion_hits == [entry.id], "a text change still re-indexes"
    assert entry.id not in tomato_hits


def test_a_saved_entry_reads_back_with_its_source(tmp_path):
    history.bootstrap(tmp_path)
    saved = history.save_entry(
        "an uploaded talk", duration_ms=1, source="file", source_name="talk.mp3"
    )

    (read,) = history.get_page().entries

    assert (read.id, read.source, read.source_name, read.starred) == (
        saved.id,
        "file",
        "talk.mp3",
        False,
    )


@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ("C:\\Users\\me\\Music\\standup.m4a", "standup.m4a"),
        ("/Users/me/Music/standup.m4a", "standup.m4a"),
        ("   ", None),
        (None, None),
        ("x" * 250 + ".mp3", "x" * schema.SOURCE_NAME_MAX),
    ],
    ids=["windows-path", "posix-path", "blank", "none", "too-long"],
)
def test_a_file_name_is_stored_as_the_name_the_user_saw(tmp_path, given, stored):
    history.bootstrap(tmp_path)
    history.save_entry("text", duration_ms=1, source="file", source_name=given)

    assert history.get_page().entries[0].source_name == stored


def test_the_store_refuses_a_source_it_does_not_know(tmp_path):
    history.bootstrap(tmp_path)
    with history._lock:
        conn = history._ensure_conn_locked()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO entries(id, ts, language, raw_text, cleaned_text, duration_ms, "
                "source) VALUES ('x', 1, 'uk', 't', 't', 0, 'podcast')"
            )


def test_a_merged_store_with_unreadable_source_values_arrives_repaired(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    raw = sqlite3.connect(source_dir / "history.db")
    try:
        raw.execute(
            "CREATE TABLE entries (id, ts, language, raw_text, cleaned_text, duration_ms, "
            "source, source_name, starred)"
        )
        raw.executemany(
            "INSERT INTO entries VALUES (?, 1700000000000, 'uk', 't', 't', 1, ?, ?, ?)",
            [
                ("kept", "meeting", "call.wav", 1),
                ("odd", "podcast", 42, 7),
                ("long", "file", "y" * 300, "yes"),
            ],
        )
        raw.commit()
    finally:
        raw.close()

    outcome, _ = relocation.consolidate_into(source_dir, tmp_path)
    assert outcome is relocation.ConsolidateOutcome.CONSOLIDATED
    history.bootstrap(tmp_path)

    by_id = {e.id: (e.source, e.source_name, e.starred) for e in history.get_page().entries}
    assert by_id == {
        "kept": ("meeting", "call.wav", True),
        "odd": ("dictation", None, False),
        "long": ("file", "y" * schema.SOURCE_NAME_MAX, False),
    }
