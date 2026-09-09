import { confirm } from "@tauri-apps/plugin-dialog";
import { api, type HistoryCursor, type HistoryEntry } from "../api";

/** The singular/plural pair a tab uses when it names its own rows. */
export interface HistoryListNoun {
  singular: string;
  plural: string;
}

/** The five elements the shared list writes to. Each tab owns its own markup and passes them in. */
export interface HistoryListElements {
  count: HTMLElement;
  rows: HTMLElement;
  loadMoreWrapper: HTMLElement;
  loadMoreButton: HTMLButtonElement;
  clearButton: HTMLButtonElement;
}

export interface HistoryListOptions {
  pageSize: number;
  noun: HistoryListNoun;
  elements: HistoryListElements;
  createRow: (entry: HistoryEntry) => HTMLElement;
  renderEmptyState: (isEmpty: boolean) => void;
  isDestroyed: () => boolean;
  onCleared?: () => void;
}

export interface HistoryList {
  /** Loads the first page, dropping any cursor, and replaces whatever is painted. */
  load(): Promise<void>;
  /** Overrides the count text — for a tab lane the list does not own, such as History's search. */
  renderCount(text: string): void;
  renderLoadMore(visible: boolean): void;
  /** Drops one from the running total after a single-entry delete. */
  entryRemoved(): void;
}

export function formatEntryCount(total: number, noun: HistoryListNoun): string {
  return `${total} ${total === 1 ? noun.singular : noun.plural}`;
}

/**
 * Pagination, failure text, "Load more" wiring and the Clear All flow for the two tabs
 * that page over `api.getHistory`. It never creates markup and never owns a row's shape.
 */
export function createHistoryList(options: HistoryListOptions): HistoryList {
  const { pageSize, noun, elements, createRow, renderEmptyState, isDestroyed, onCleared } = options;

  let cursor: HistoryCursor | null = null;
  let total = 0;
  let pageSeq = 0;
  let pageInFlight = false;

  function renderCount(text: string): void {
    elements.count.textContent = text;
  }

  function renderLoadMore(visible: boolean): void {
    elements.loadMoreWrapper.style.display = visible ? "block" : "none";
  }

  /**
   * One page, appended or replacing. A request that has been superseded -- by a
   * reload, by Clear All, or by a second "Load more" -- paints nothing and leaves
   * the cursor alone, so whichever request is current decides both, rather than
   * whichever answered last.
   *
   * `next_cursor` is normalised at the edge because `request` casts the response
   * rather than validating it: a backend that predates the cursor contract omits
   * the field, and `undefined !== null` would leave "Load more" visible forever
   * with every click re-fetching and re-appending the first page.
   */
  async function loadPage(append: boolean): Promise<void> {
    const seq = ++pageSeq;
    pageInFlight = true;
    try {
      const response = await api.getHistory(pageSize, cursor);
      if (isDestroyed() || seq !== pageSeq) return;

      total = response.total;
      renderCount(formatEntryCount(total, noun));

      if (!append) {
        elements.rows.innerHTML = "";
      }

      for (const entry of response.entries) {
        elements.rows.appendChild(createRow(entry));
      }

      renderEmptyState(response.entries.length === 0 && !append);

      const nextCursor = response.next_cursor ?? null;
      cursor = nextCursor;
      renderLoadMore(nextCursor !== null);
    } catch (error) {
      if (isDestroyed() || seq !== pageSeq) return;
      renderCount("Failed to load");
      console.error(error);
    } finally {
      if (seq === pageSeq) {
        pageInFlight = false;
      }
    }
  }

  async function clearAll(): Promise<void> {
    if (total === 0) return;
    elements.clearButton.disabled = true;
    const confirmed = await confirm(
      `Delete all ${formatEntryCount(total, noun)}? History and Metrics share the same data — both tabs will be cleared.`,
      { title: "Clear History", kind: "warning" }
    );
    if (!confirmed) {
      if (!isDestroyed()) {
        elements.clearButton.disabled = false;
      }
      return;
    }
    if (!isDestroyed()) {
      elements.clearButton.textContent = "Clearing...";
    }
    try {
      await api.clearHistory();
      if (isDestroyed()) return;
      ++pageSeq;
      cursor = null;
      total = 0;
      elements.rows.innerHTML = "";
      renderEmptyState(true);
      renderCount(formatEntryCount(0, noun));
      renderLoadMore(false);
      onCleared?.();
    } catch (error) {
      console.error(error);
    } finally {
      if (!isDestroyed()) {
        elements.clearButton.disabled = false;
        elements.clearButton.textContent = "Clear All";
      }
    }
  }

  elements.loadMoreButton.addEventListener("click", () => {
    if (pageInFlight) return;
    void loadPage(true);
  });

  elements.clearButton.addEventListener("click", () => {
    void clearAll();
  });

  return {
    load() {
      cursor = null;
      return loadPage(false);
    },
    renderCount,
    renderLoadMore,
    entryRemoved() {
      total--;
      renderCount(formatEntryCount(total, noun));
    },
  };
}
