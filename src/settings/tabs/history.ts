import { api, type HistoryEntry } from "../../api";

const DATE_FMT = new Intl.DateTimeFormat("uk-UA", {
  day: "2-digit",
  month: "short",
});
const TIME_FMT = new Intl.DateTimeFormat("uk-UA", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export function renderHistory(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2 class="tab-title">History</h2>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <span class="value" id="history-count">Loading...</span>
      <button class="btn btn-danger" id="btn-clear-history">Clear All</button>
    </div>
    <div id="history-list"></div>
    <div id="history-load-more" style="text-align: center; padding: 12px; display: none;">
      <button class="btn btn-secondary" id="btn-load-more">Load more</button>
    </div>
  `;

  const countEl = container.querySelector<HTMLElement>("#history-count")!;
  const listEl = container.querySelector<HTMLElement>("#history-list")!;
  const loadMoreWrap = container.querySelector<HTMLElement>("#history-load-more")!;
  const btnLoadMore = container.querySelector<HTMLButtonElement>("#btn-load-more")!;
  const btnClear = container.querySelector<HTMLButtonElement>("#btn-clear-history")!;

  let offset = 0;
  let total = 0;
  const LIMIT = 30;

  async function loadEntries(append = false) {
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

    el.innerHTML = `
      <div class="history-entry-header">
        <div class="history-stamp">
          <span class="history-stamp-date">${DATE_FMT.format(date)}</span>
          <span class="history-stamp-time">${TIME_FMT.format(date)}</span>
        </div>
        <div class="history-badges">${badges.join("")}</div>
      </div>
      <div class="history-text">${escapeHtml(entry.cleaned_text)}</div>
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
        await navigator.clipboard.writeText(entry.raw_text || entry.cleaned_text);
        target.textContent = "Copied!";
        setTimeout(() => (target.textContent = "Copy"), 1500);
      } else if (action === "delete") {
        try {
          await api.deleteHistoryEntry(entry.id);
          el.remove();
          total--;
          countEl.textContent = `${total} transcript${total !== 1 ? "s" : ""}`;
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

  return () => {};
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}
