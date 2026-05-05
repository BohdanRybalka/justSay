import { api, type HistoryStats } from "../../api";

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

export function renderWords(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Words</h2>
    <div id="words-body">
      <div class="value" id="words-loading">Loading...</div>
    </div>
  `;

  const body = container.querySelector<HTMLElement>("#words-body")!;

  let cancelled = false;

  async function refresh() {
    try {
      const s = await api.historyStats();
      if (cancelled) return;
      body.innerHTML = renderStats(s);
    } catch (e) {
      if (cancelled) return;
      body.innerHTML = `<div class="value" style="color:var(--red)">Failed to load: ${(e as Error).message}</div>`;
    }
  }

  refresh();
  const poll = setInterval(refresh, 5000);

  return () => {
    cancelled = true;
    clearInterval(poll);
  };
}

function renderStats(s: HistoryStats): string {
  if (s.total_entries === 0) {
    return `
      <div class="value" style="color:var(--text-muted); padding:32px 0; text-align:center;">
        No transcriptions yet. Dictate something, then come back.
      </div>
    `;
  }

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

  const langBlock = renderBucket("By language", s.by_language, (code) => LANGUAGE_LABELS[code] || code);
  const modelBlock = renderBucket("By model", s.by_model, (m) => m);

  return `
    ${cards}
    ${audioBlock}
    ${langBlock}
    ${modelBlock}
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
