import {
  api,
  type HistoryStats,
  type InsightsResponse,
  type TopWordsResponse,
} from "../../api";
import { escapeHtml } from "../html";

const LANGUAGE_LABELS: Record<string, string> = {
  uk: "Ukrainian",
  en: "English",
  de: "German",
  fr: "French",
  es: "Spanish",
  pl: "Polish",
  ja: "Japanese",
  zh: "Chinese",
  auto: "Auto-detected",
};

type Lang = "all" | "uk" | "en";
const TOP_LIMIT = 10;

export function renderWords(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Words</h2>
    <div id="words-body">
      <div class="value" id="words-loading">Loading...</div>
    </div>
  `;

  const body = container.querySelector<HTMLElement>("#words-body")!;

  let cancelled = false;
  let topLang: Lang = "all";
  // Concurrent-call guard: only one /words/insights at a time.
  let insightsLoading = false;
  let insightsLoadedOnce = false;
  // /words/top + /words/insights are post-v0.10.4 — older frozen sidecars
  // return 404; remember that so we degrade gracefully.
  let topUnsupported = false;
  let insightsUnsupported = false;
  // True once renderPage has assigned the full body markup; refreshStats
  // can skip its work while we're still in the zero-state placeholder.
  let pageRendered = false;
  let lastTotalEntries = -1;

  function isNotFound(e: unknown): boolean {
    const msg = (e as Error).message?.toLowerCase() ?? "";
    return (
      msg.includes("not found") ||
      msg.includes("http 404") ||
      msg.includes("method not allowed") ||
      msg.includes("http 405")
    );
  }

  async function fetchTop(): Promise<TopWordsResponse | null> {
    if (topUnsupported) return null;
    try {
      return await api.wordsTop(topLang, TOP_LIMIT);
    } catch (e) {
      if (isNotFound(e)) {
        topUnsupported = true;
      } else {
        console.error("wordsTop failed:", e);
      }
      return null;
    }
  }

  async function renderPage() {
    try {
      const stats = await api.historyStats();
      if (cancelled) return;
      const top = await fetchTop();
      if (cancelled) return;
      lastTotalEntries = stats.total_entries;

      body.innerHTML = renderBody(stats, top, topLang);
      pageRendered = stats.total_entries > 0;

      if (pageRendered) {
        if (top) wireLangToggle();
        if (top && !insightsUnsupported && !insightsLoading && !insightsLoadedOnce) {
          loadInsights(false);
        }
      }
    } catch (e) {
      if (cancelled) return;
      body.innerHTML = `<div class="value" style="color:var(--red)">Failed to load: ${(e as Error).message}</div>`;
      pageRendered = false;
    }
  }

  async function refreshStats() {
    try {
      const stats = await api.historyStats();
      if (cancelled) return;

      // First tick before the initial renderPage finished — skip without
      // synthesising a fake "zero → non-zero" transition that would race
      // an in-flight renderPage(), produce a detached insights box, and
      // strand it on "Loading insights…".
      if (lastTotalEntries < 0) return;

      // Zero ↔ non-zero transition triggers a full re-paint instead of
      // mutating absent nodes.
      const wasEmpty = lastTotalEntries === 0;
      const isEmpty = stats.total_entries === 0;
      lastTotalEntries = stats.total_entries;
      if (wasEmpty !== isEmpty) {
        await renderPage();
        return;
      }
      if (isEmpty || !pageRendered) return;

      const top = await fetchTop();
      if (cancelled) return;

      setText("words-stat-today", stats.today_words.toLocaleString("uk-UA"));
      setText("words-stat-week", stats.week_words.toLocaleString("uk-UA"));
      setText("words-stat-lifetime", stats.total_words.toLocaleString("uk-UA"));
      setText("words-stat-audio", formatDuration(stats.total_audio_seconds));
      setText("words-stat-entries", stats.total_entries.toLocaleString("uk-UA"));

      if (top) {
        const topEl = document.getElementById("words-top");
        if (topEl) topEl.innerHTML = renderTopWords(top);
      }

      const langEl = document.getElementById("words-by-lang");
      if (langEl) {
        langEl.innerHTML = renderBucketRows(stats.by_language, (code) => LANGUAGE_LABELS[code] || code);
      }
      const modelEl = document.getElementById("words-by-model");
      if (modelEl) {
        modelEl.innerHTML = renderBucketRows(stats.by_model, (m) => m);
      }
    } catch (e) {
      if (cancelled) return;
      console.error("refreshStats failed:", e);
    }
  }

  function setText(id: string, value: string) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function wireLangToggle() {
    const toggle = body.querySelector<HTMLElement>("#words-lang-toggle");
    if (!toggle) return;
    toggle.querySelectorAll<HTMLButtonElement>("button").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const next = btn.dataset.lang as Lang | undefined;
        if (!next || next === topLang) return;
        topLang = next;
        try {
          const top = await api.wordsTop(topLang, TOP_LIMIT);
          if (cancelled) return;
          const topContainer = body.querySelector<HTMLElement>("#words-top");
          if (topContainer) topContainer.innerHTML = renderTopWords(top);
          toggle.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
            b.classList.toggle("active", b.dataset.lang === topLang);
          });
          // Auto-refresh insights to keep the two sections in sync. Only
          // fires after the top-words update succeeded — if wordsTop failed
          // we keep both sections on their previous content. Fire-and-forget;
          // the `insightsLoading` guard inside loadInsights() blocks
          // overlapping requests when the user spams the toggle.
          if (!insightsUnsupported) {
            void loadInsights(true);
          }
        } catch (e) {
          console.error(e);
        }
      });
    });
  }

  async function loadInsights(force: boolean) {
    // Race-condition guard — stays active regardless of `force`.
    if (insightsLoading) return;
    if (!force && insightsLoadedOnce) return;
    const box = body.querySelector<HTMLElement>("#words-insights-box");
    if (!box) return;
    insightsLoading = true;
    box.innerHTML = `<div class="value" style="color:var(--text-muted)">Generating insights...</div>`;
    try {
      const insights = await api.wordsInsights();
      if (cancelled) return;
      box.innerHTML = renderInsights(insights);
      insightsLoadedOnce = true;
    } catch (e) {
      if (cancelled) return;
      if (isNotFound(e)) {
        insightsUnsupported = true;
        // Drop the whole insights block silently — older sidecar.
        const wrap = body.querySelector<HTMLElement>("#words-insights-wrap");
        if (wrap) wrap.remove();
        return;
      }
      const msg = (e as Error).message || "Insights unavailable";
      box.innerHTML = `
        <div class="value" style="color:var(--text-muted)">
          Insights unavailable — start Ollama or switch to Cloud mode.
          <div style="font-size: 11px; opacity: 0.7; margin-top: 4px">${escapeHtml(msg)}</div>
        </div>`;
      // Do NOT set insightsLoadedOnce — error states are retryable on the
      // next language-toggle click (which calls loadInsights(true)).
    } finally {
      insightsLoading = false;
    }
  }

  renderPage();
  const poll = setInterval(refreshStats, 5000);

  return () => {
    cancelled = true;
    clearInterval(poll);
  };
}

function renderBody(
  stats: HistoryStats,
  top: TopWordsResponse | null,
  lang: Lang,
): string {
  if (stats.total_entries === 0) {
    return `
      <div class="value" style="color:var(--text-muted); padding:32px 0; text-align:center;">
        No transcriptions yet. Dictate something, then come back.
      </div>
    `;
  }

  return `
    ${renderStatsCards(stats)}
    ${top ? renderTopWordsBlock(top, lang) : ""}
    ${top ? renderInsightsBlock() : ""}
    ${renderBucket("By language", "words-by-lang", stats.by_language, (code) => LANGUAGE_LABELS[code] || code)}
    ${renderBucket("By model", "words-by-model", stats.by_model, (m) => m)}
  `;
}

function renderStatsCards(s: HistoryStats): string {
  const cards = `
    <div class="word-cards">
      ${bigCard("Today", "words-stat-today", s.today_words)}
      ${bigCard("This week", "words-stat-week", s.week_words)}
      ${bigCard("Lifetime", "words-stat-lifetime", s.total_words)}
    </div>
  `;

  const audioBlock = `
    <div class="setting-row" style="margin-top:16px;">
      <span class="label">Total audio time</span>
      <span class="value" id="words-stat-audio">${formatDuration(s.total_audio_seconds)}</span>
    </div>
    <div class="setting-row">
      <span class="label">Transcriptions</span>
      <span class="value" id="words-stat-entries">${s.total_entries.toLocaleString("uk-UA")}</span>
    </div>
  `;

  return cards + audioBlock;
}

function renderTopWordsBlock(top: TopWordsResponse, lang: Lang): string {
  const langButtons = (["all", "uk", "en"] as Lang[])
    .map(
      (k) => `
        <button class="btn btn-secondary btn-sm ${k === lang ? "active" : ""}" data-lang="${k}">
          ${k === "all" ? "All" : LANGUAGE_LABELS[k] || k}
        </button>`,
    )
    .join("");

  return `
    <div class="setting-group" style="margin-top:24px;">
      <div class="setting-label" style="display:flex; justify-content:space-between; align-items:center;">
        <span>Top words</span>
        <span id="words-lang-toggle" style="display:flex; gap:6px;">${langButtons}</span>
      </div>
      <div id="words-top">${renderTopWords(top)}</div>
    </div>
  `;
}

function renderTopWords(top: TopWordsResponse): string {
  if (top.items.length === 0) {
    return `<div class="value" style="color:var(--text-muted); padding:16px 0;">No words yet for this filter.</div>`;
  }
  const max = top.items[0].count || 1;
  const rows = top.items
    .map((item) => {
      const pct = (item.count / max) * 100;
      return `
        <div class="bucket-row">
          <span class="bucket-label">${escapeHtml(item.word)}</span>
          <div class="bucket-bar"><div class="bucket-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
          <span class="bucket-value">${item.count.toLocaleString("uk-UA")}</span>
        </div>
      `;
    })
    .join("");

  return `
    <div class="bucket-list">${rows}</div>
    <div class="value" style="color:var(--text-muted); font-size:11px; margin-top:6px;">
      Based on ${top.scanned.toLocaleString("uk-UA")} transcript${top.scanned !== 1 ? "s" : ""}.
    </div>
  `;
}

function renderInsightsBlock(): string {
  return `
    <div class="setting-group" id="words-insights-wrap" style="margin-top:24px;">
      <div class="setting-label">Insights</div>
      <div id="words-insights-box">
        <div class="value" style="color:var(--text-muted)">Loading insights...</div>
      </div>
    </div>
  `;
}

function renderInsights(payload: InsightsResponse): string {
  if (payload.insights.length === 0) {
    return `<div class="value" style="color:var(--text-muted)">No insights yet — dictate a few transcripts and come back.</div>`;
  }
  const cards = payload.insights
    .map(
      (text) => `
        <div class="word-card" style="text-align:left; padding:10px 12px;">
          <div class="word-card-sub" style="font-size:13px; color:var(--text);">
            ${escapeHtml(text)}
          </div>
        </div>`,
    )
    .join("");
  return `
    <div class="word-cards" style="grid-template-columns: 1fr;">${cards}</div>
    <div class="value" style="color:var(--text-muted); font-size:11px; margin-top:6px;">
      Generated by ${escapeHtml(payload.model)}.
    </div>
  `;
}

function bigCard(label: string, valueId: string, count: number): string {
  return `
    <div class="word-card">
      <div class="word-card-label">${label}</div>
      <div class="word-card-value" id="${valueId}">${count.toLocaleString("uk-UA")}</div>
      <div class="word-card-sub">words</div>
    </div>
  `;
}

function renderBucketRows(
  bucket: Record<string, number>,
  labelFn: (k: string) => string,
): string {
  const entries = Object.entries(bucket).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return "";
  const max = entries[0][1] || 1;
  return entries
    .map(([key, value]) => {
      const pct = (value / max) * 100;
      return `
        <div class="bucket-row">
          <span class="bucket-label">${escapeHtml(labelFn(key))}</span>
          <div class="bucket-bar"><div class="bucket-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
          <span class="bucket-value">${value.toLocaleString("uk-UA")}</span>
        </div>
      `;
    })
    .join("");
}

function renderBucket(
  title: string,
  id: string,
  bucket: Record<string, number>,
  labelFn: (k: string) => string,
): string {
  // Always emit the container even when empty so refreshStats() can rely
  // on the IDs being present from first mount.
  return `
    <div class="setting-group" style="margin-top:20px;">
      <div class="setting-label">${title}</div>
      <div class="bucket-list" id="${id}">${renderBucketRows(bucket, labelFn)}</div>
    </div>
  `;
}

function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return "0 m";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h} h ${m} m`;
  if (m > 0) return `${m} m ${s} s`;
  return `${s} s`;
}

