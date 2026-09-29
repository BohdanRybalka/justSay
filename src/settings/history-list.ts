import {
  api,
  SidecarTooOldError,
  type HistoryCursor,
  type HistoryDay,
  type HistoryEntry,
  type HistoryPageResponse,
} from "../api";
import { isStaleStatusResponse } from "../stale-response";

/** The singular/plural pair a tab uses when it names its own rows. */
export interface HistoryListNoun {
  singular: string;
  plural: string;
}

/** A row the list built from an entry before anything on screen was touched. */
export interface BuiltRow {
  entry: HistoryEntry;
  element: HTMLElement;
}

/**
 * Where rows go. The tab owns the layout — day groups, headers — and the list
 * owns when and in what order a page lands.
 */
export interface HistoryRows {
  /** Replaces whatever is painted with the first page, newest first. */
  replace(rows: readonly BuiltRow[], days: readonly HistoryDay[]): void;
  /** Adds an older page below what is painted, newest first. */
  append(rows: readonly BuiltRow[], days: readonly HistoryDay[]): void;
  /** Adds rows newer than everything painted above it, newest first. */
  prepend(rows: readonly BuiltRow[], days: readonly HistoryDay[]): void;
  /** Paints another lane's elements in place of the list's own. */
  replaceWith(elements: readonly HTMLElement[]): void;
}

/**
 * The two elements the shared list writes to. `sentinel` sits below the rows:
 * scrolling it into view asks for the next page, and while a page is read it
 * carries the `--reading` class the skeleton cards hang on.
 */
export interface HistoryListElements {
  count: HTMLElement;
  sentinel: HTMLElement;
}

export interface HistoryListOptions {
  pageSize: number;
  noun: HistoryListNoun;
  /** The tab's own name, as the version-skew message says it. */
  featureName: string;
  elements: HistoryListElements;
  rows: HistoryRows;
  createRow: (entry: HistoryEntry) => HTMLElement;
  renderEmptyState: (isEmpty: boolean) => void;
  isDestroyed: () => boolean;
  /** Read on every request, so a reload after the filter changes pages under the new one. */
  starredOnly: () => boolean;
}

export const SENTINEL_READING = "timeline-more--reading";

/**
 * A lane's permission to write to the shared count and sentinel, valid only
 * while nothing newer has taken the rows over.
 *
 * Taking a claim supersedes every other lane and holds paging: while the latest
 * claim is unreleased, the sentinel asks for nothing and the newer-rows poll
 * skips its turn, so nothing pages under a paint that has not happened yet.
 * `release()` ends the hold and looks at the sentinel again.
 *
 * `replaceRows` is the only way to paint the rows from outside this module, and
 * painting through it records that the rows on screen are no longer the list's
 * own page, which is what stops paging and the poll until a reload.
 */
export interface HistoryRowsClaim {
  isCurrent(): boolean;
  renderCount(text: string): void;
  renderMore(visible: boolean): void;
  replaceRows(rows: readonly HTMLElement[]): void;
  release(): void;
}

export interface HistoryList {
  /**
   * Asks for the first page with no cursor and, once it arrives, replaces
   * whatever is painted and adopts both cursors the response carried. A request
   * that fails changes nothing.
   */
  load(): Promise<void>;
  /**
   * Asks for the rows newer than the newest one painted and puts them on top,
   * and keeps asking while pages come back full. Does nothing while another
   * request is outstanding or the rows belong to another lane; a failure is
   * logged and paints nothing, since the next turn asks again.
   */
  loadNewer(): Promise<void>;
  /**
   * Takes the rows over for a lane the list does not own, such as History's
   * search, and hands back the only way to write to the shared count and
   * sentinel from outside this module. The lane calls `release()` when it is
   * done, and a lane that painted its own rows hides the sentinel itself.
   */
  claimRows(): HistoryRowsClaim;
  /**
   * Drops one from the running total after a single-entry delete, and does
   * nothing at all while the rows on screen belong to another lane.
   */
  entryRemoved(): void;
  /** Stops watching the sentinel. */
  disconnect(): void;
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
 * Pagination, the newer-rows read, failure text and the sentinel for a tab that
 * pages over `api.getHistory`. It never creates markup and never owns a row's
 * shape or its place on screen.
 */
export function createHistoryList(options: HistoryListOptions): HistoryList {
  const {
    pageSize,
    noun,
    featureName,
    elements,
    rows,
    createRow,
    renderEmptyState,
    isDestroyed,
    starredOnly,
  } = options;

  let cursor: HistoryCursor | null = null;
  let newest: HistoryCursor | null = null;
  let total = 0;
  let latestIssuedToken = 0;
  let backendOmitsCursor = false;
  let latestClaim: HistoryRowsClaim | null = null;
  let rowsAreOwnPage = false;
  let pagingHeld = false;

  const observer = new IntersectionObserver((records) => {
    if (records.some((record) => record.isIntersecting)) void loadOlder();
  });
  observer.observe(elements.sentinel);

  /**
   * Observing again is what makes the observer report where the sentinel is
   * now: a page that landed may or may not have pushed it out of view, and only
   * a fresh observation says which after layout.
   */
  function rewatchSentinel(): void {
    observer.unobserve(elements.sentinel);
    observer.observe(elements.sentinel);
  }

  /**
   * The total, or the version-skew warning while the backend is still omitting
   * the cursor. Sticky against `entryRemoved`; a page that comes back carrying a
   * cursor is what clears it.
   */
  function renderTotal(claim: HistoryRowsClaim): void {
    claim.renderCount(
      backendOmitsCursor ? sidecarTooOldText(featureName) : formatEntryCount(total, noun)
    );
  }

  /**
   * Issues a claim, which supersedes every lane already holding one: the token
   * is bumped, the new claim is recorded as the list's own writer and paging is
   * held for the duration of the lane. Calling it "just to get a claim"
   * silently invalidates every request in flight.
   */
  function issueClaim(): HistoryRowsClaim {
    const token = ++latestIssuedToken;
    const isCurrent = () => !isDestroyed() && !isStaleStatusResponse(token, latestIssuedToken);
    const claim: HistoryRowsClaim = {
      isCurrent,
      renderCount(text: string) {
        if (isCurrent()) elements.count.textContent = text;
      },
      renderMore(visible: boolean) {
        if (isCurrent()) elements.sentinel.hidden = !visible;
      },
      replaceRows(painted: readonly HTMLElement[]) {
        if (!isCurrent()) return;
        rows.replaceWith(painted);
        rowsAreOwnPage = false;
      },
      release() {
        if (!isCurrent()) return;
        pagingHeld = false;
        elements.sentinel.classList.remove(SENTINEL_READING);
        rewatchSentinel();
      },
    };
    latestClaim = claim;
    if (isCurrent()) pagingHeld = true;
    return claim;
  }

  function build(entries: readonly HistoryEntry[]): BuiltRow[] {
    return entries.map((entry) => ({ entry, element: createRow(entry) }));
  }

  /**
   * One page, appended or replacing. A reload asks with no cursor without
   * discarding the stored one, and every row is built before the count is
   * written or anything is painted, so a request that fails, is superseded, or
   * carries an entry `createRow` cannot build leaves the rows and both cursors
   * exactly as they were.
   */
  async function loadPage(append: boolean): Promise<void> {
    if (append && cursor === null) return;
    const claim = issueClaim();
    elements.sentinel.classList.add(SENTINEL_READING);
    try {
      const response = await api.getHistory(pageSize, append ? cursor : null, starredOnly());
      if (!claim.isCurrent()) return;

      const built = build(response.entries);

      total = response.total;
      backendOmitsCursor = false;
      rowsAreOwnPage = true;
      renderTotal(claim);

      if (append) {
        rows.append(built, response.days);
      } else {
        rows.replace(built, response.days);
        newest = response.newest_cursor;
      }
      renderEmptyState(response.entries.length === 0 && !append);

      cursor = response.next_cursor;
      claim.renderMore(cursor !== null);
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

  function loadOlder(): Promise<void> {
    if (pagingHeld || !rowsAreOwnPage) return Promise.resolve();
    return loadPage(true);
  }

  async function loadNewer(): Promise<void> {
    if (isDestroyed() || pagingHeld || !rowsAreOwnPage) return;
    if (newest === null) return loadPage(false);
    const claim = issueClaim();
    let page: HistoryPageResponse | null = null;
    try {
      page = await api.getNewerHistory(pageSize, newest, starredOnly());
      if (!claim.isCurrent()) return;
      const built = build([...page.entries].reverse());
      total = page.total;
      renderTotal(claim);
      if (built.length > 0) rows.prepend(built, page.days);
      newest = page.newest_cursor ?? newest;
    } catch (error) {
      page = null;
      console.error(error);
    } finally {
      claim.release();
    }
    if (page !== null && page.entries.length === pageSize) await loadNewer();
  }

  return {
    load() {
      return loadPage(false);
    },
    loadNewer,
    claimRows: issueClaim,
    entryRemoved() {
      if (!rowsAreOwnPage || latestClaim === null) return;
      total--;
      renderTotal(latestClaim);
    },
    disconnect() {
      observer.disconnect();
    },
  };
}
