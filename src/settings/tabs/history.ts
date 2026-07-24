import { api, type HistoryEntry } from "../../api";
import { escapeHtml } from "../html";

const DATE_FMT = new Intl.DateTimeFormat("uk-UA", {
  day: "2-digit",
  month: "short",
});
const TIME_FMT = new Intl.DateTimeFormat("uk-UA", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const SEARCH_DEBOUNCE_MS = 300;

export function renderHistory(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2 class="tab-title">History</h2>
    <div style="margin-bottom: 12px;">
      <input
        type="search"
        id="history-search"
        placeholder="Search transcripts..."
        style="width: 100%; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text); font-size: 14px;"
      />
      <div id="history-search-hint" style="font-size: 11px; color: var(--text-muted); margin-top: 4px; min-height: 14px;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <span class="value" id="history-count">Loading...</span>
      <button class="btn btn-danger" id="btn-clear-history">Clear All</button>
    </div>
    <div id="history-list"></div>
    <div id="history-load-more" style="text-align: center; padding: 12px; display: none;">
      <button class="btn btn-secondary" id="btn-load-more">Load more</button>
    </div>
  `;

  const searchInput = container.querySelector<HTMLInputElement>("#history-search")!;
  const searchHint = container.querySelector<HTMLElement>("#history-search-hint")!;
  const countEl = container.querySelector<HTMLElement>("#history-count")!;
  const listEl = container.querySelector<HTMLElement>("#history-list")!;
  const loadMoreWrap = container.querySelector<HTMLElement>("#history-load-more")!;
  const btnLoadMore = container.querySelector<HTMLButtonElement>("#btn-load-more")!;
  const btnClear = container.querySelector<HTMLButtonElement>("#btn-clear-history")!;

  let offset = 0;
  let total = 0;
  const LIMIT = 30;
  let inSearchMode = false;
  let debounceTimer: number | null = null;
  let searchSeq = 0;
  let destroyed = false;

  async function loadEntries(append = false) {
    inSearchMode = false;
    ++searchSeq;
    try {
      const resp = await api.getHistory(LIMIT, offset);
      total = resp.total;
      countEl.textContent = `${total} transcript${total !== 1 ? "s" : ""}`;

      if (!append) {
        listEl.innerHTML = "";
      }

      if (resp.entries.length === 0 && !append) {
        listEl.innerHTML = `<div style="color: var(--text-muted); padding: 32px; text-align: center;">No transcripts yet</div>`;
      }

      for (const entry of resp.entries) {
        listEl.appendChild(createEntryEl(entry));
      }

      offset += resp.entries.length;
      loadMoreWrap.style.display = offset < total ? "block" : "none";
    } catch (e) {
      countEl.textContent = "Failed to load";
      console.error(e);
    }
  }

  async function runSearch(q: string) {
    inSearchMode = true;
    const seq = ++searchSeq;
    searchHint.textContent = "Searching...";
    try {
      const resp = await api.searchHistory(q, LIMIT);
      if (seq !== searchSeq) return;
      listEl.innerHTML = "";
      countEl.textContent = `${resp.total} match${resp.total !== 1 ? "es" : ""}`;
      if (resp.entries.length === 0) {
        listEl.innerHTML = `<div style="color: var(--text-muted); padding: 32px; text-align: center;">No matches</div>`;
      }
      for (const entry of resp.entries) {
        listEl.appendChild(createEntryEl(entry));
      }
      loadMoreWrap.style.display = "none";
      searchHint.textContent = "";
    } catch (e) {
      if (seq !== searchSeq) return;
      const msg = (e as Error).message || "Search failed";
      const lower = msg.toLowerCase();
      const sidecarTooOld =
        lower.includes("not found") ||
        lower.includes("http 404") ||
        lower.includes("method not allowed") ||
        lower.includes("http 405");
      if (sidecarTooOld) {
        searchHint.textContent = "Search needs the latest backend — please update JustSay.";
      } else {
        searchHint.textContent = lower.includes("invalid")
          ? "Invalid search query"
          : msg;
      }
    }
  }

  searchInput.addEventListener("input", () => {
    if (debounceTimer !== null) {
      window.clearTimeout(debounceTimer);
    }
    const value = searchInput.value.trim();
    debounceTimer = window.setTimeout(() => {
      debounceTimer = null;
      if (!value) {
        offset = 0;
        searchHint.textContent = "";
        loadEntries(false);
      } else {
        runSearch(value);
      }
    }, SEARCH_DEBOUNCE_MS);
  });

  function createEntryEl(entry: HistoryEntry): HTMLElement {
    const el = document.createElement("div");
    el.className = "history-entry";
    el.dataset.id = entry.id;

    const date = new Date(entry.timestamp);

    const badges: string[] = [];
    badges.push(`<span class="history-badge">${(entry.duration_ms / 1000).toFixed(2)} s process</span>`);
    if (entry.audio_duration_seconds != null) {
      badges.push(`<span class="history-badge">${entry.audio_duration_seconds.toFixed(1)} s audio</span>`);
    }
    if (entry.word_count != null) {
      badges.push(`<span class="history-badge">${entry.word_count} words</span>`);
    }
    badges.push(`<span class="history-badge">${escapeHtml(entry.language)}</span>`);
    if (entry.style === "ai_prompt") {
      badges.push(`<span class="history-badge history-badge-ai">AI Prompt</span>`);
    }

    const textHtml = entry.highlighted_text
      ? entry.highlighted_text
      : escapeHtml(entry.text).replace(/\n/g, "<br>");

    el.innerHTML = `
      <div class="history-entry-header">
        <div class="history-stamp">
          <span class="history-stamp-date">${DATE_FMT.format(date)}</span>
          <span class="history-stamp-time">${TIME_FMT.format(date)}</span>
        </div>
        <div class="history-badges">${badges.join("")}</div>
      </div>
      <div class="history-text">${textHtml}</div>
      <div class="history-actions">
        <button class="btn btn-secondary btn-sm" data-action="copy">Copy</button>
        <button class="btn btn-secondary btn-sm" data-action="delete">Delete</button>
      </div>
    `;

    el.addEventListener("click", async (e) => {
      const target = e.target as HTMLElement;
      const action = target.dataset.action;
      if (!action) return;

      if (action === "copy") {
        await navigator.clipboard.writeText(entry.text);
        target.textContent = "Copied!";
        setTimeout(() => (target.textContent = "Copy"), 1500);
      } else if (action === "delete") {
        try {
          await api.deleteHistoryEntry(entry.id);
          el.remove();
          if (!inSearchMode) {
            total--;
            countEl.textContent = `${total} transcript${total !== 1 ? "s" : ""}`;
          }
        } catch (err) {
          console.error(err);
        }
      }
    });

    return el;
  }

  btnLoadMore.addEventListener("click", () => loadEntries(true));

  btnClear.addEventListener("click", async () => {
    if (total === 0) return;
    btnClear.disabled = true;
    btnClear.textContent = "Clearing...";
    try {
      await api.clearHistory();
      offset = 0;
      total = 0;
      searchInput.value = "";
      inSearchMode = false;
      listEl.innerHTML = `<div style="color: var(--text-muted); padding: 32px; text-align: center;">No transcripts yet</div>`;
      countEl.textContent = "0 transcripts";
      loadMoreWrap.style.display = "none";
    } catch (e) {
      console.error(e);
    } finally {
      btnClear.disabled = false;
      btnClear.textContent = "Clear All";
    }
  });

  loadEntries();

  return () => {
    destroyed = true;
    if (debounceTimer !== null) window.clearTimeout(debounceTimer);
  };
}

