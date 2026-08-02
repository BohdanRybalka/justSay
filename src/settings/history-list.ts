import { confirm } from "@tauri-apps/plugin-dialog";
import { api, type HistoryEntry } from "../api";

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
  /** Loads the first page from offset 0, replacing whatever is painted. */
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

  let offset = 0;
  let total = 0;

  function renderCount(text: string): void {
    elements.count.textContent = text;
  }

  function renderLoadMore(visible: boolean): void {
    elements.loadMoreWrapper.style.display = visible ? "block" : "none";
  }

  async function loadPage(append: boolean): Promise<void> {
    try {
      const response = await api.getHistory(pageSize, offset);
      if (isDestroyed()) return;

      total = response.total;
      renderCount(formatEntryCount(total, noun));

      if (!append) {
        elements.rows.innerHTML = "";
      }

      for (const entry of response.entries) {
        elements.rows.appendChild(createRow(entry));
      }

      renderEmptyState(response.entries.length === 0 && !append);

      offset += response.entries.length;
      renderLoadMore(offset < total);
    } catch (error) {
      if (isDestroyed()) return;
      renderCount("Failed to load");
      console.error(error);
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
      offset = 0;
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
      offset = 0;
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
