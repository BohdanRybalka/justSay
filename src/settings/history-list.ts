import { confirm } from "@tauri-apps/plugin-dialog";
import { api, type HistoryCursor, type HistoryEntry } from "../api";
import { isStaleStatusResponse } from "../stale-response";

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
  /**
   * Asks for the first page with no cursor and, once it arrives, replaces
   * whatever is painted and adopts the cursor the response carried. A request
   * that fails changes nothing.
   */
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
  let latestIssuedToken = 0;

  function renderCount(text: string): void {
    elements.count.textContent = text;
  }

  function renderLoadMore(visible: boolean): void {
    elements.loadMoreWrapper.style.display = visible ? "block" : "none";
  }

  /**
   * One page, appended or replacing. A reload asks with no cursor without
   * discarding the stored one: nothing painted is destroyed before its
   * replacement has arrived, so a request that fails or is superseded leaves the
   * rows, the cursor and "Load more" exactly as they were. A reload whose
   * request 503s would otherwise leave the button visible over a null cursor,
   * and the next click would re-fetch and re-append the first page -- the
   * duplicate this whole spec exists to remove.
   *
   * Staleness is `isStaleStatusResponse`, the one mechanism this app uses for a
   * late answer, and a second click is refused by disabling the button rather
   * than by a flag beside it: the condition then lives on the element it
   * governs and cannot drift out of step with a counter.
   *
   * `next_cursor` is normalised at the edge because `request` casts the response
   * rather than validating it: a backend that predates the cursor contract omits
   * the field, and `undefined !== null` would leave "Load more" visible forever.
   */
  async function loadPage(append: boolean): Promise<void> {
    const token = ++latestIssuedToken;
    elements.loadMoreButton.disabled = true;
    try {
      const response = await api.getHistory(pageSize, append ? cursor : null);
      if (isDestroyed() || isStaleStatusResponse(token, latestIssuedToken)) return;

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
      if (isDestroyed() || isStaleStatusResponse(token, latestIssuedToken)) return;
      renderCount("Failed to load");
      console.error(error);
    } finally {
      if (!isDestroyed() && !isStaleStatusResponse(token, latestIssuedToken)) {
        elements.loadMoreButton.disabled = false;
      }
    }
  }

  /**
   * Deletes everything and puts the list back in its opening state. It supersedes
   * any outstanding page without issuing a request of its own, so it re-enables
   * the button itself -- the superseded `loadPage` sees a newer token and will
   * not.
   */
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
      ++latestIssuedToken;
      elements.loadMoreButton.disabled = false;
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
    void loadPage(true);
  });

  elements.clearButton.addEventListener("click", () => {
    void clearAll();
  });

  return {
    load() {
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
