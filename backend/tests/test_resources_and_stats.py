"""Resource endpoint extensions and history aggregation."""

from datetime import datetime, timedelta, timezone

import pytest

from app.core import router as core_router
from app.core.gpu_probe import GpuProbeResult, GpuVendor
from app.transcripts import history


@pytest.mark.asyncio
async def test_resources_includes_cpu_percent_and_gb(client):
    """`/resources` must expose process CPU% and GB-formatted RAM."""
    resp = await client.get("/resources")
    assert resp.status_code == 200
    data = resp.json()

    assert data["cpu_percent_process"] >= 0
    assert data["cpu_percent_total"] >= 0

    assert data["ram_total_gb"] > 0
    assert data["pid_ram_gb"] >= 0
    assert abs(data["ram_total_gb"] - data["ram_total_mb"] / 1024) < 0.01
    assert abs(data["pid_ram_gb"] - data["pid_ram_mb"] / 1024) < 0.01




def test_get_gpu_info_returns_none_when_probe_reports_none(monkeypatch):
    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(vendor=GpuVendor.NONE),
    )
    assert core_router._get_gpu_info() is None


def test_get_gpu_info_nvidia_populates_used_and_free(monkeypatch):
    """torch.cuda-shaped result: name/vendor/total/used/free all populated."""
    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(
            vendor=GpuVendor.NVIDIA, name="RTX 4090",
            vram_total_mb=24576, vram_used_mb=2048, vram_free_mb=22528,
        ),
    )
    info = core_router._get_gpu_info()
    assert info is not None
    assert info.name == "RTX 4090"
    assert info.vendor == "nvidia"
    assert info.vram_total_mb == 24576
    assert info.vram_used_mb == 2048
    assert info.vram_free_mb == 22528


def test_get_gpu_info_amd_reports_total_only(monkeypatch):
    """Registry-shaped AMD result has no live-usage reading — used/free both None."""
    monkeypatch.setattr(
        "app.core.gpu_probe.probe_gpu",
        lambda: GpuProbeResult(
            vendor=GpuVendor.AMD, name="AMD Radeon RX 5700 XT", vram_total_mb=8172,
        ),
    )
    info = core_router._get_gpu_info()
    assert info is not None
    assert info.name == "AMD Radeon RX 5700 XT"
    assert info.vendor == "amd"
    assert info.vram_total_mb == 8172
    assert info.vram_used_mb is None
    assert info.vram_free_mb is None


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
    assert stats.week_words == 150
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
