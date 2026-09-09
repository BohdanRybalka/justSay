import type { HistoryCursor, HistoryEntry, HistoryPageResponse } from "../api";

/** One history row, distinguished only by its id — the field every list test asserts on. */
export function buildEntry(id: string): HistoryEntry {
  return {
    id,
    timestamp: "2026-08-01T10:00:00Z",
    language: "uk",
    style: "normal",
    text: `transcript ${id}`,
    duration_ms: 1200,
    model_name: "whisper",
    tokens_used: null,
    audio_duration_seconds: 3.5,
    word_count: 4,
  };
}

/**
 * A stand-in for `api.getHistory` that serves the page the cursor names, found by
 * identity rather than by counting how many rows it has already handed out. A null
 * cursor is the first page; a cursor is the rows strictly after the entry it names.
 *
 * A stub that counts cannot tell a client that echoes `next_cursor` from one that
 * sends `null` every time, and answers a reset cursor with page 2 — which is the
 * trap spec 134's Risks section names and cites spec 126 for.
 */
export function pagesByCursor(
  all: HistoryEntry[]
): (limit: number, cursor: HistoryCursor | null) => Promise<HistoryPageResponse> {
  return async (limit: number, cursor: HistoryCursor | null) => {
    const start = cursor === null ? 0 : all.findIndex((entry) => entry.id === cursor.id) + 1;
    const entries = all.slice(start, start + limit);
    const last = entries[entries.length - 1];
    const end = start + entries.length;
    return {
      entries,
      total: all.length,
      next_cursor: last && end < all.length ? { ts: end, id: last.id } : null,
    };
  };
}
