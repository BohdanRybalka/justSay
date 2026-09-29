import type { HistoryCursor, HistoryEntry, HistoryPageResponse } from "../api";

/** One history row, distinguished only by its id — the field every list test asserts on. */
export function buildEntry(id: string): HistoryEntry {
  return {
    id,
    timestamp: "2026-08-01T10:00:00Z",
    language: "uk",
    text: `transcript ${id}`,
    duration_ms: 1200,
    model_name: "whisper",
    tokens_used: null,
    audio_duration_seconds: 3.5,
    word_count: 4,
    source: "dictation",
    source_name: null,
    starred: false,
  };
}

/** A position naming `entry`, the way the backend mints one. */
export function positionOf(entry: HistoryEntry): HistoryCursor {
  return { ts: 0, id: entry.id };
}

/** A whole `/history` answer; `newest_cursor` names the first entry unless given. */
export function pageOf(
  entries: HistoryEntry[],
  total: number,
  next_cursor: HistoryCursor | null,
  extra: Partial<HistoryPageResponse> = {},
): HistoryPageResponse {
  return {
    entries,
    total,
    next_cursor,
    newest_cursor: entries.length > 0 ? positionOf(entries[0]) : null,
    days: [],
    ...extra,
  };
}

/**
 * A stand-in for `api.getHistory` that serves the page the cursor names, found by
 * identity rather than by counting how many rows it has already handed out. A null
 * cursor is the first page; a cursor is the rows strictly after the entry it names.
 *
 * A stub that counts cannot tell a client that echoes `next_cursor` from one that
 * sends `null` every time, and answers a reset cursor with page 2.
 */
export function pagesByCursor(
  all: HistoryEntry[]
): (limit: number, cursor: HistoryCursor | null) => Promise<HistoryPageResponse> {
  return async (limit: number, cursor: HistoryCursor | null) => {
    const start = cursor === null ? 0 : all.findIndex((entry) => entry.id === cursor.id) + 1;
    const entries = all.slice(start, start + limit);
    const last = entries[entries.length - 1];
    const end = start + entries.length;
    return pageOf(entries, all.length, last && end < all.length ? { ts: end, id: last.id } : null);
  };
}

/**
 * A stand-in for `api.getNewerHistory` over `all`, newest first and free to grow:
 * the rows before the one `after` names, oldest first, at most `limit` of them.
 */
export function newerByCursor(
  all: HistoryEntry[]
): (limit: number, after: HistoryCursor) => Promise<HistoryPageResponse> {
  return async (limit: number, after: HistoryCursor) => {
    const index = all.findIndex((entry) => entry.id === after.id);
    const entries = all.slice(0, index).reverse().slice(0, limit);
    const newest = entries[entries.length - 1];
    return pageOf(entries, all.length, null, {
      newest_cursor: newest ? positionOf(newest) : null,
    });
  };
}

/** An `IntersectionObserver` a test drives by hand: `cross()` reports the sentinel in view. */
export class FakeObserver {
  static latest: FakeObserver | null = null;
  observeCalls = 0;
  watching = false;

  constructor(private readonly callback: IntersectionObserverCallback) {
    FakeObserver.latest = this;
  }

  observe(): void {
    this.observeCalls += 1;
    this.watching = true;
  }

  unobserve(): void {
    this.watching = false;
  }

  disconnect(): void {
    this.watching = false;
  }

  cross(): void {
    this.callback(
      [{ isIntersecting: true } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver
    );
  }
}

/** A `ResizeObserver` a test drives by hand: `resize()` reports every watched element changed. */
export class FakeResizeObserver {
  static latest: FakeResizeObserver | null = null;
  private readonly targets: Element[] = [];

  constructor(private readonly callback: ResizeObserverCallback) {
    FakeResizeObserver.latest = this;
  }

  observe(target: Element): void {
    this.targets.push(target);
  }

  unobserve(): void {}

  disconnect(): void {
    this.targets.length = 0;
  }

  resize(): void {
    this.callback(
      this.targets.map((target) => ({ target }) as ResizeObserverEntry),
      this as unknown as ResizeObserver
    );
  }
}
