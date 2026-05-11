import {
  api,
  type HistoryStats,
  type InsightsResponse,
  type TopWordsResponse,
} from "../../api";

const LANGUAGE_LABELS: Record<string, string> = {
  uk: "Ukrainian",
  en: "English",
  de: "German",
  fr: "French",
  es: "Spanish",
  pl: "Polish",
  ja: "Japanese",
  zh: "Chinese",
};

type Lang = "all" | "uk" | "en";

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
  // Guard against the 5 s polling firing a new insights LLM call while
  // the previous one is still in flight. Without this, a slow Ollama
  // (10–30 s) produces a fetch storm — each in-flight /words/insights
  // holds history._lock through the top_words SQL, racing dictation
  // for the same lock. Closes QA exit-gate RED-2.
  let insightsLoading = false;
  let insightsLoadedOnce = false;
  // The /words/top + /words/insights endpoints are post-v0.10.4. Older
  // frozen sidecars return 404; remember that so we stop probing on every
  // 5 s poll and degrade gracefully to the legacy stats-only view.
  let topUnsupported = false;
  let insightsUnsupported = false;

  async function refresh() {
    try {
      const stats = await api.historyStats();
      if (cancelled) return;

      let top: TopWordsResponse | null = null;
      if (!topUnsupported) {
        try {
          top = await api.wordsTop(topLang, 30);
        } catch (e) {
          if (isNotFound(e)) {
            topUnsupported = true;
          } else {
            console.error("wordsTop failed:", e);
          }
        }
      }
      if (cancelled) return;

      body.innerHTML = renderBody(stats, top, topLang, topUnsupported || insightsUnsupported);
      if (top) wireLangToggle();

      // Insights load lazily after the main paint — they involve an LLM call.
      // Only kick off a fetch if the endpoint is reachable, no other call is
      // in flight, and we haven't already painted insights this tab open.
      if (insightsUnsupported) {
        const box = body.querySelector<HTMLElement>("#words-insights-box");
        if (box) box.remove();
      } else if (!insightsLoading && !insightsLoadedOnce) {
        loadInsights();
      } else if (insightsLoadedOnce) {
        const box = body.querySelector<HTMLElement>("#words-insights-box");
        if (box) {
          box.innerHTML = `<div class="value" style="color:var(--text-muted)">Insights cached — open this tab again to refresh.</div>`;
        }
      }
    } catch (e) {
      if (cancelled) return;
      body.innerHTML = `<div class="value" style="color:var(--red)">Failed to load: ${(e as Error).message}</div>`;
    }
  }

  function isNotFound(e: unknown): boolean {
    const msg = (e as Error).message?.toLowerCase() ?? "";
    // 404 (route missing) or 405 (FastAPI matches an existing path with a
    // different verb — happens because the old sidecar exposed only
    // DELETE /history/{entry_id}, so any GET to a new sibling path gets
    // method-not-allowed). Both mean: "sidecar is too old, hide the block."
    return (
      msg.includes("not found") ||
      msg.includes("http 404") ||
      msg.includes("method not allowed") ||
      msg.includes("http 405")
    );
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
          const top = await api.wordsTop(topLang, 30);
          if (cancelled) return;
          const topContainer = body.querySelector<HTMLElement>("#words-top");
          if (topContainer) topContainer.innerHTML = renderTopWords(top);
          toggle.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
            b.classList.toggle("active", b.dataset.lang === topLang);
          });
        } catch (e) {
          console.error(e);
        }
      });
    });
  }

  async function loadInsights() {
    if (insightsLoading) return;
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
        // Older sidecar — feature not available. Drop the block silently
        // and stop probing on subsequent polls.
        insightsUnsupported = true;
        box.remove();
        return;
      }
      const msg = (e as Error).message || "Insights unavailable";
      box.innerHTML = `
        <div class="value" style="color:var(--text-muted)">
          Insights unavailable — start Ollama or switch to Cloud mode.
          <div style="font-size: 11px; opacity: 0.7; margin-top: 4px">${escapeHtml(msg)}</div>
        </div>`;
      // Do NOT set insightsLoadedOnce — a failure should be retried by
      // the next refresh, NOT cached as a permanent error state.
    } finally {
      insightsLoading = false;
    }
  }

  refresh();
  // History stats poll every 5 s (existing behaviour). Top-words and
  // insights cache for 1 h on the backend; refresh handles cache hits.
  const poll = setInterval(refresh, 5000);

  return () => {
    cancelled = true;
    clearInterval(poll);
  };
}

function renderBody(
  stats: HistoryStats,
  top: TopWordsResponse | null,
  lang: Lang,
  _legacyOnly: boolean,
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
    ${renderBucket("By language", stats.by_language, (code) => LANGUAGE_LABELS[code] || code)}
    ${renderBucket("By model", stats.by_model, (m) => m)}
  `;
}

function renderStatsCards(s: HistoryStats): string {
  const cards = `
    <div class="word-cards">
      ${bigCard("Today", s.today_words)}
      ${bigCard("This week", s.week_words)}
      ${bigCard("Lifetime", s.total_words)}
    </div>
  `;

  const audioBlock = `
    <div class="setting-row" style="margin-top:16px;">
      <span class="label">Total audio time</span>
      <span class="value">${formatDuration(s.total_audio_seconds)}</span>
    </div>
    <div class="setting-row">
      <span class="label">Transcriptions</span>
      <span class="value">${s.total_entries.toLocaleString("uk-UA")}</span>
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
    <div class="setting-group" style="margin-top:24px;">
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

function bigCard(label: string, count: number): string {
  return `
    <div class="word-card">
      <div class="word-card-label">${label}</div>
      <div class="word-card-value">${count.toLocaleString("uk-UA")}</div>
      <div class="word-card-sub">words</div>
    </div>
  `;
}

function renderBucket(title: string, bucket: Record<string, number>, labelFn: (k: string) => string): string {
  const entries = Object.entries(bucket).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return "";
  const max = entries[0][1] || 1;

  const rows = entries.map(([key, value]) => {
    const pct = (value / max) * 100;
    return `
      <div class="bucket-row">
        <span class="bucket-label">${escapeHtml(labelFn(key))}</span>
        <div class="bucket-bar"><div class="bucket-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="bucket-value">${value.toLocaleString("uk-UA")}</span>
      </div>
    `;
  }).join("");

  return `
    <div class="setting-group" style="margin-top:20px;">
      <div class="setting-label">${title}</div>
      <div class="bucket-list">${rows}</div>
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

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
