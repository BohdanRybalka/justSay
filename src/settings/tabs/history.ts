import { api, SidecarTooOldError, type HistoryEntry } from "../../api";
import { createHistoryList, sidecarTooOldText, type HistoryRowsClaim } from "../history-list";
import { escapeHtml } from "../html";

const DATE_FORMATTER = new Intl.DateTimeFormat("uk-UA", {
  day: "2-digit",
  month: "short",
});
const TIME_FORMATTER = new Intl.DateTimeFormat("uk-UA", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

const SEARCH_DEBOUNCE_MS = 300;
const PAGE_SIZE = 30;
const EMPTY_HTML = `<div style="color: var(--text-muted); padding: 32px; text-align: center;">No transcripts yet</div>`;

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

  let searchClaim: HistoryRowsClaim | null = null;
  let debounceTimer: number | null = null;
  let destroyed = false;

  const list = createHistoryList({
    pageSize: PAGE_SIZE,
    noun: { singular: "transcript", plural: "transcripts" },
    featureName: "History",
    elements: {
      count: countEl,
      rows: listEl,
      loadMoreWrapper: loadMoreWrap,
      loadMoreButton: btnLoadMore,
      clearButton: btnClear,
    },
    createRow: createEntryElement,
    renderEmptyState: (isEmpty) => {
      if (isEmpty) listEl.innerHTML = EMPTY_HTML;
    },
    isDestroyed: () => destroyed,
    onCleared: () => {
      searchInput.value = "";
      searchHint.textContent = "";
    },
  });

  /**
   * The search lane, holding a claim on the shared rows for as long as what is on
   * screen is its own paint.
   *
   * The claim is taken before the request rather than after it, so a page load
   * already in flight is superseded at the moment the user asks for matches and
   * cannot repaint the unfiltered history over them when it answers. It is what
   * "am I showing search results?" reads, so the lane that painted and the lane
   * the tab believes painted are always the same lane.
   *
   * The hint is the one element the claim does not cover, because it belongs to
   * this tab rather than to the shared list. A search that is itself superseded
   * returns without writing its own outcome, so the `finally` clears the hint it
   * put up -- otherwise "Searching..." stays on screen for good over rows some
   * other lane painted.
   *
   * What the `finally` asks is whether a newer *search* exists, not whether this
   * lane is still current. A newer search owns the hint and has already written
   * its own text into it, so blanking it there would erase a live error message
   * -- the same defect, moved one element over. Only a reload or a Clear All can
   * supersede this lane while `searchClaim` still points at it, and after one of
   * those there is no search on screen for the hint to describe.
   */
  async function runSearch(q: string) {
    const claim = list.claimRows();
    searchClaim = claim;
    searchHint.textContent = "Searching...";
    try {
      const resp = await api.searchHistory(q, PAGE_SIZE);
      if (!claim.isCurrent()) return;
      listEl.innerHTML = "";
      claim.renderCount(`${resp.total} match${resp.total !== 1 ? "es" : ""}`);
      if (resp.entries.length === 0) {
        listEl.innerHTML = `<div style="color: var(--text-muted); padding: 32px; text-align: center;">No matches</div>`;
      }
      for (const entry of resp.entries) {
        listEl.appendChild(createEntryElement(entry));
      }
      claim.renderLoadMore(false);
      searchHint.textContent = "";
    } catch (e) {
      if (!claim.isCurrent()) return;
      if (e instanceof SidecarTooOldError) {
        searchHint.textContent = sidecarTooOldText("Search");
      } else {
        const msg = (e as Error).message || "Search failed";
        searchHint.textContent = msg.toLowerCase().includes("invalid")
          ? "Invalid search query"
          : msg;
      }
    } finally {
      claim.release();
      if (!destroyed && searchClaim === claim && !claim.isCurrent()) {
        searchHint.textContent = "";
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
        searchHint.textContent = "";
        void list.load();
      } else {
        void runSearch(value);
      }
    }, SEARCH_DEBOUNCE_MS);
  });

  function createEntryElement(entry: HistoryEntry): HTMLElement {
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
          <span class="history-stamp-date">${DATE_FORMATTER.format(date)}</span>
          <span class="history-stamp-time">${TIME_FORMATTER.format(date)}</span>
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
          list.entryRemoved();
        } catch (err) {
          console.error(err);
        }
      }
    });

    return el;
  }

  void list.load();

  return () => {
    destroyed = true;
    if (debounceTimer !== null) window.clearTimeout(debounceTimer);
  };
}
