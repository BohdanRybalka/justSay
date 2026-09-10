import { confirm } from "@tauri-apps/plugin-dialog";
import { api, SidecarTooOldError, type HistoryCursor, type HistoryEntry } from "../api";
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
  /** The tab's own name, as the version-skew message says it. */
  featureName: string;
  elements: HistoryListElements;
  createRow: (entry: HistoryEntry) => HTMLElement;
  renderEmptyState: (isEmpty: boolean) => void;
  isDestroyed: () => boolean;
  onCleared?: () => void;
}

/**
 * A lane's permission to write to the shared count and "Load more" elements,
 * valid only while nothing newer has taken the rows over.
 *
 * Taking a claim is what supersedes every other lane, so one counter decides
 * both who paints and who the tab believes painted. Both writers are no-ops
 * once `isCurrent()` is false, so a late answer needs no staleness test of its
 * own beyond the one it already asks before touching rows.
 *
 * `release()` hands "Load more" back once the lane is done with it. Taking a
 * claim disables the button rather than hiding it, so nothing can page under a
 * paint that has not happened yet; releasing is how the button comes back for a
 * lane that ended without painting rows of its own.
 *
 * `replaceRows` is the only way to paint the shared row container from outside
 * this module, and painting through it is what records that the rows on screen
 * are no longer the list's own page. A lane that claims and then never answers
 * therefore leaves the list's own rows described by the list's own count.
 */
export interface HistoryRowsClaim {
  isCurrent(): boolean;
  renderCount(text: string): void;
  renderLoadMore(visible: boolean): void;
  replaceRows(rows: readonly HTMLElement[]): void;
  release(): void;
}

export interface HistoryList {
  /**
   * Asks for the first page with no cursor and, once it arrives, replaces
   * whatever is painted and adopts the cursor the response carried. A request
   * that fails changes nothing.
   */
  load(): Promise<void>;
  /**
   * Takes the rows over for a lane the list does not own, such as History's
   * search, and hands back the only way to write to the shared count and
   * "Load more" from outside this module.
   *
   * Claiming disables "Load more" for as long as the lane runs and leaves the
   * wrapper alone. The claiming lane is about to paint something the stored
   * cursor does not describe, so a click while it runs would append rows from a
   * page nothing on screen came from; a disabled button refuses that click
   * without taking the button away from rows the lane may never repaint. The
   * lane calls `release()` when it is done, and a lane that painted its own
   * rows hides the wrapper itself.
   *
   * The hold lasts exactly as long as the claiming lane is outstanding, and
   * nothing bounds that from here. `api.searchHistory` has no client-side
   * budget, so a backend that never answers a search leaves the button disabled
   * until some newer lane supersedes the claim -- emptying the search box, which
   * reloads the page and releases the button, is the recovery.
   */
  claimRows(): HistoryRowsClaim;
  /**
   * Drops one from the running total after a single-entry delete, and does
   * nothing at all while the rows on screen belong to another lane.
   *
   * The list knows whose paint is showing, so the caller does not have to: a
   * delete during a History search leaves the match count alone, and the same
   * delete after a reload has taken the rows back decrements the total. Asking
   * the caller to guard meant asking a claim whether it was superseded, which
   * is a question a claim answers `false` to after teardown as well.
   */
  entryRemoved(): void;
}

/**
 * The one shape both version-skew messages are built from, so a wording change
 * cannot reach the history one and miss the search one.
 */
export function sidecarTooOldText(feature: string): string {
  return `${feature} needs the latest backend — please update JustSay.`;
}

export function formatEntryCount(total: number, noun: HistoryListNoun): string {
  return `${total} ${total === 1 ? noun.singular : noun.plural}`;
}

/**
 * Pagination, failure text, "Load more" wiring and the Clear All flow for the two tabs
 * that page over `api.getHistory`. It never creates markup and never owns a row's shape.
 */
export function createHistoryList(options: HistoryListOptions): HistoryList {
  const { pageSize, noun, featureName, elements, createRow, renderEmptyState, isDestroyed, onCleared } =
    options;

  let cursor: HistoryCursor | null = null;
  let total = 0;
  let latestIssuedToken = 0;
  let backendOmitsCursor = false;
  let latestClaim: HistoryRowsClaim | null = null;
  let rowsAreOwnPage = false;

  function renderCount(text: string): void {
    elements.count.textContent = text;
  }

  function renderLoadMore(visible: boolean): void {
    elements.loadMoreWrapper.style.display = visible ? "block" : "none";
  }

  /**
   * The total, or the version-skew warning while the backend is still omitting
   * the cursor.
   *
   * Sticky against `entryRemoved`, not permanent: the count is the number that
   * is lying when the backend omits the cursor, so deleting a row afterwards
   * must not paint a plausible total back over the warning. A page that comes
   * back carrying a cursor is what clears it, so restarting the backend on a
   * version that answers correctly stops the warning without a remount.
   */
  function renderTotal(claim: HistoryRowsClaim): void {
    claim.renderCount(
      backendOmitsCursor ? sidecarTooOldText(featureName) : formatEntryCount(total, noun)
    );
  }

  /**
   * Issues a claim, which supersedes every lane already holding one: the token
   * is bumped, the new claim is recorded as the list's own writer and
   * "Load more" is disabled for the duration of the lane. Reading this as a
   * plain factory is the mistake it is named against -- calling it "just to get
   * a claim" silently invalidates every request in flight.
   *
   * The disable lives here rather than in each caller because both of them go
   * through this one function, and the one place that knows whether the tab is
   * still mounted is the `isCurrent` this claim is built around.
   */
  function issueClaim(): HistoryRowsClaim {
    const token = ++latestIssuedToken;
    const isCurrent = () => !isDestroyed() && !isStaleStatusResponse(token, latestIssuedToken);
    const claim: HistoryRowsClaim = {
      isCurrent,
      renderCount(text: string) {
        if (isCurrent()) renderCount(text);
      },
      renderLoadMore(visible: boolean) {
        if (isCurrent()) renderLoadMore(visible);
      },
      replaceRows(rows: readonly HTMLElement[]) {
        if (!isCurrent()) return;
        elements.rows.innerHTML = "";
        for (const row of rows) {
          elements.rows.appendChild(row);
        }
        rowsAreOwnPage = false;
      },
      release() {
        if (isCurrent()) elements.loadMoreButton.disabled = false;
      },
    };
    latestClaim = claim;
    if (isCurrent()) {
      elements.loadMoreButton.disabled = true;
    }
    return claim;
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
   * Staleness is `isStaleStatusResponse` behind a claim, the one mechanism this
   * app uses for a late answer, and a second click is refused by disabling the
   * button rather than by a flag beside it: the condition then lives on the
   * element it governs and cannot drift out of step with a counter.
   *
   * An append with no stored cursor would ask for the first page and append it
   * under itself, so it is refused before the claim is issued and before the
   * button is touched: a refused click supersedes nothing and leaves the button
   * as it found it. That invariant used to be held up by every call site
   * happening to hide the button, which the next call site added would not have
   * known to do.
   */
  async function loadPage(append: boolean): Promise<void> {
    if (append && cursor === null) return;
    const claim = issueClaim();
    try {
      const response = await api.getHistory(pageSize, append ? cursor : null);
      if (!claim.isCurrent()) return;

      total = response.total;
      backendOmitsCursor = false;
      rowsAreOwnPage = true;
      renderTotal(claim);

      if (!append) {
        elements.rows.innerHTML = "";
      }

      for (const entry of response.entries) {
        elements.rows.appendChild(createRow(entry));
      }

      renderEmptyState(response.entries.length === 0 && !append);

      cursor = response.next_cursor;
      claim.renderLoadMore(cursor !== null);
    } catch (error) {
      if (!claim.isCurrent()) return;
      if (error instanceof SidecarTooOldError) {
        backendOmitsCursor = true;
        renderTotal(claim);
      } else {
        claim.renderCount("Failed to load");
      }
      console.error(error);
    } finally {
      claim.release();
    }
  }

  /**
   * Deletes everything and puts the list back in its opening state. It supersedes
   * any outstanding page without issuing a request of its own, so it releases the
   * button itself once it has painted -- the superseded `loadPage` sees a newer
   * token and will not.
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
      const claim = issueClaim();
      cursor = null;
      total = 0;
      elements.rows.innerHTML = "";
      renderEmptyState(true);
      rowsAreOwnPage = true;
      renderTotal(claim);
      claim.renderLoadMore(false);
      claim.release();
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
    claimRows: issueClaim,
    entryRemoved() {
      if (!rowsAreOwnPage || latestClaim === null) return;
      total--;
      renderTotal(latestClaim);
    },
  };
}
