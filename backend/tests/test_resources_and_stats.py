"""Resource endpoint extensions and history aggregation."""

from datetime import datetime, timedelta, timezone

import pytest

from app.core import history


@pytest.mark.asyncio
async def test_resources_includes_cpu_percent_and_gb(client):
    """`/resources` must expose process CPU% and GB-formatted RAM."""
    resp = await client.get("/resources")
    assert resp.status_code == 200
    data = resp.json()

    # Process & system CPU percentage fields exist and are non-negative.
    assert data["cpu_percent_process"] >= 0
    assert data["cpu_percent_total"] >= 0

    # GB fields are positive floats and match the MB versions within rounding.
    assert data["ram_total_gb"] > 0
    assert data["pid_ram_gb"] >= 0
    assert abs(data["ram_total_gb"] - data["ram_total_mb"] / 1024) < 0.01
    assert abs(data["pid_ram_gb"] - data["pid_ram_mb"] / 1024) < 0.01


def _insert_entry(words: int, lang: str, model: str, when: datetime) -> None:
    """Direct DB insert with a controlled timestamp (save_entry uses utcnow)."""
    ts_ms = int(round(when.timestamp() * 1000))
    with history._lock:
        conn = history._ensure_conn_locked()
        conn.execute(
            "INSERT INTO entries(id, ts, language, style, raw_text, cleaned_text, "
            "duration_ms, audio_duration_seconds, word_count, model_name, tokens_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"id-{words}-{lang}-{model}", ts_ms, lang, "normal",
                "x " * words, "x " * words, 1000,
                5.0, words, model, None,
            ),
        )
        history._stats_cache = None


@pytest.fixture(autouse=True)
def _close_history_conn():
    yield
    with history._lock:
        history._close_conn_locked()


def test_compute_stats_buckets_by_language_and_model(tmp_path, monkeypatch):
    """`compute_stats` must total words and split by language + model."""
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    now = datetime.now(timezone.utc).astimezone()
    yesterday = now - timedelta(days=1)
    last_month = now - timedelta(days=30)

    _insert_entry(100, "uk", "gemini-2.5-flash", now)
    _insert_entry(50, "en", "groq/whisper-large-v3-turbo", yesterday)
    _insert_entry(200, "uk", "groq/whisper-large-v3-turbo", last_month)

    stats = history.compute_stats(now=now)

    assert stats.total_entries == 3
    assert stats.total_words == 350
    assert stats.today_words == 100
    assert stats.week_words == 150  # today + yesterday only; the month-old one is out
    assert stats.by_language == {"uk": 300, "en": 50}
    assert stats.by_model == {
        "gemini-2.5-flash": 100,
        "groq/whisper-large-v3-turbo": 250,
    }
    assert stats.total_audio_seconds == 15.0

    with history._lock:
        history._close_conn_locked()


@pytest.mark.asyncio
async def test_history_stats_endpoint(client, tmp_path, monkeypatch):
    """`GET /history/stats` returns the same payload as `compute_stats`."""
    monkeypatch.setattr(history, "_output_dir", tmp_path)
    monkeypatch.setattr(history, "_conn", None)
    monkeypatch.setattr(history, "_stats_cache", None)

    now = datetime.now(timezone.utc).astimezone()
    _insert_entry(42, "uk", "gemini-2.5-flash", now)

    resp = await client.get("/history/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_entries"] == 1
    assert data["total_words"] == 42
    assert data["by_language"] == {"uk": 42}

    with history._lock:
        history._close_conn_locked()
