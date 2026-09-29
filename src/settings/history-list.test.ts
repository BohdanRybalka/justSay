// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryCursor, HistoryDay, HistoryEntry, HistoryPageResponse } from "../api";
import type { BuiltRow, HistoryRows } from "./history-list";
import {
  buildEntry,
  FakeObserver,
  newerByCursor,
  pageOf,
  positionOf,
} from "./history-page-stub.test-helper";

const apiMock = {
  getHistory: vi.fn(),
  getNewerHistory: vi.fn(),
};

/**
 * Only `api` is replaced. Everything else in the module -- `SidecarTooOldError`
 * above all -- stays the real export, so the `instanceof` branch under test is
 * tied to the class `api.getHistory` actually throws.
 */
vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

const { SidecarTooOldError } = await import("../api");
const { api: unmockedApi } = await vi.importActual<typeof import("../api")>("../api");
const { createHistoryList, formatEntryCount, sidecarTooOldText, SENTINEL_READING } = await import(
  "./history-list"
);

const TRANSCRIPTS = { singular: "transcript", plural: "transcripts" };
const ENTRIES = { singular: "entry", plural: "entries" };

function cross(): void {
  FakeObserver.latest!.cross();
}

describe("formatEntryCount — counts in the noun it is given", () => {
  it("uses the plural for zero", () => {
    expect(formatEntryCount(0, TRANSCRIPTS)).toBe("0 transcripts");
    expect(formatEntryCount(0, ENTRIES)).toBe("0 entries");
  });

  it("uses the singular for exactly one", () => {
    expect(formatEntryCount(1, TRANSCRIPTS)).toBe("1 transcript");
    expect(formatEntryCount(1, ENTRIES)).toBe("1 entry");
  });

  it("uses the plural for two", () => {
    expect(formatEntryCount(2, TRANSCRIPTS)).toBe("2 transcripts");
    expect(formatEntryCount(2, ENTRIES)).toBe("2 entries");
  });
});

interface Harness {
  elements: { count: HTMLElement; sentinel: HTMLElement };
  rows: HistoryRows;
  placedDays: HistoryDay[][];
  paintedIds: () => string[];
  sentinelVisible: () => boolean;
  reading: () => boolean;
  countText: () => string;
}

/** A flat stand-in for the tab's timeline: rows in one container, in the order placed. */
function harness(): Harness {
  const count = document.createElement("div");
  const sentinel = document.createElement("div");
  const container = document.createElement("div");
  const placedDays: HistoryDay[][] = [];
  const elementsOf = (rows: readonly BuiltRow[]) => rows.map((row) => row.element);
  const rows: HistoryRows = {
    replace(built, days) {
      placedDays.push([...days]);
      container.replaceChildren(...elementsOf(built));
    },
    append(built, days) {
      placedDays.push([...days]);
      container.append(...elementsOf(built));
    },
    prepend(built, days) {
      placedDays.push([...days]);
      container.prepend(...elementsOf(built));
    },
    replaceWith(elements) {
      container.replaceChildren(...elements);
    },
  };
  return {
    elements: { count, sentinel },
    rows,
    placedDays,
    paintedIds: () => Array.from(container.children).map((el) => el.textContent!),
    sentinelVisible: () => !sentinel.hidden,
    reading: () => sentinel.classList.contains(SENTINEL_READING),
    countText: () => count.textContent!,
  };
}

function listOver(
  h: Harness,
  pageSize = 2,
  isDestroyed: () => boolean = () => false,
  createRow: (entry: HistoryEntry) => HTMLElement = defaultCreateRow
) {
  return createHistoryList({
    pageSize,
    noun: TRANSCRIPTS,
    featureName: "History",
    elements: h.elements,
    rows: h.rows,
    createRow,
    renderEmptyState: () => {},
    isDestroyed,
  });
}

function defaultCreateRow(entry: HistoryEntry): HTMLElement {
  const row = document.createElement("div");
  row.textContent = entry.id;
  return row;
}

/** What the tab's real card builder does with an entry missing a field it reads. */
function createRowRequiringText(entry: HistoryEntry): HTMLElement {
  if (typeof entry.text !== "string") {
    throw new TypeError("entry.text is not a string");
  }
  return defaultCreateRow(entry);
}

/**
 * A backend the test drives by hand. Each queued response is handed out in
 * order; what is asserted is the cursor the list *sent*.
 */
function queueResponses(...responses: HistoryPageResponse[]): void {
  let index = 0;
  apiMock.getHistory.mockImplementation(async () => responses[index++]);
}

function sentCursors(): (HistoryCursor | null)[] {
  return apiMock.getHistory.mock.calls.map((call) => call[1] as HistoryCursor | null);
}

function entries(...ids: string[]): HistoryEntry[] {
  return ids.map(buildEntry);
}

/**
 * A response the list is still waiting on. `release` hands it the page, so a test
 * can drive what happens while a request is outstanding.
 */
function deferredPage(): {
  promise: Promise<HistoryPageResponse>;
  release: (page: HistoryPageResponse) => void;
} {
  let release: (page: HistoryPageResponse) => void = () => {};
  const promise = new Promise<HistoryPageResponse>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("IntersectionObserver", FakeObserver);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createHistoryList — the client echoes cursors and never builds one", () => {
  it("sends no cursor for the first page", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 1, null));

    await listOver(h).load();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
    expect(apiMock.getHistory.mock.calls[0]).toEqual([2, null]);
  });

  it("sends the previous response's next_cursor back verbatim when the sentinel is crossed", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 1_700_000_000_042, id: "ff00ff00ff00" };
    queueResponses(pageOf(entries("a", "b"), 3, cursor), pageOf(entries("c"), 3, null));

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));

    expect(sentCursors()).toEqual([null, cursor]);
    await vi.waitFor(() => expect(h.paintedIds()).toEqual(["a", "b", "c"]));
  });

  it("keeps paging from the newest cursor rather than the first one", async () => {
    const h = harness();
    const first: HistoryCursor = { ts: 300, id: "second-row" };
    const second: HistoryCursor = { ts: 100, id: "fourth-row" };
    queueResponses(
      pageOf(entries("a", "b"), 5, first),
      pageOf(entries("c", "d"), 5, second),
      pageOf(entries("e"), 5, null)
    );

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(h.paintedIds()).toHaveLength(4));
    cross();
    await vi.waitFor(() => expect(h.paintedIds()).toHaveLength(5));

    expect(sentCursors()).toEqual([null, first, second]);
    expect(h.paintedIds()).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("load() asks from the newest row again without discarding the stored cursor", async () => {
    const h = harness();
    const first: HistoryCursor = { ts: 300, id: "second-row" };
    const second: HistoryCursor = { ts: 100, id: "fourth-row" };
    queueResponses(
      pageOf(entries("a", "b"), 5, first),
      pageOf(entries("c", "d"), 5, second),
      pageOf(entries("a", "b"), 5, first)
    );

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(h.paintedIds()).toHaveLength(4));
    await list.load();

    expect(sentCursors()).toEqual([null, first, null]);
    expect(h.paintedIds()).toEqual(["a", "b"]);
  });

  it("asks for nothing when the sentinel is crossed on the last page", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 1, null));

    await listOver(h).load();
    cross();
    await flush();

    expect(sentCursors()).toEqual([null]);
  });

  it("asks for nothing when the sentinel is crossed before the first page has landed", async () => {
    const h = harness();
    const first = deferredPage();
    apiMock.getHistory.mockReturnValue(first.promise);

    const list = listOver(h);
    void list.load();
    cross();
    await flush();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
  });

  it("hands every page's days to the rows it paints", async () => {
    const h = harness();
    const today: HistoryDay = { date: "2026-08-01", recordings: 9, words: 120 };
    const older: HistoryDay = { date: "2026-07-30", recordings: 1, words: 4 };
    queueResponses(
      pageOf(entries("a", "b"), 3, { ts: 1, id: "b" }, { days: [today] }),
      pageOf(entries("c"), 3, null, { days: [older] })
    );

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(h.placedDays).toHaveLength(2));

    expect(h.placedDays).toEqual([[today], [older]]);
  });
});

describe("createHistoryList — the sentinel", () => {
  it("shows the skeleton while a page is read and takes it away when the page lands", async () => {
    const h = harness();
    const second = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 1 ? pageOf(entries("a", "b"), 3, { ts: 1, id: "b" }) : second.promise;
    });

    const list = listOver(h);
    await list.load();
    expect(h.reading()).toBe(false);

    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    expect(h.reading()).toBe(true);

    second.release(pageOf(entries("c"), 3, null));
    await flush();

    expect(h.reading()).toBe(false);
  });

  it("looks at the sentinel afresh once a page has landed, so a sentinel still in view pages again", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a", "b"), 3, { ts: 1, id: "b" }));

    await listOver(h).load();

    expect(FakeObserver.latest!.observeCalls).toBe(2);
    expect(FakeObserver.latest!.watching).toBe(true);
  });

  it("stops watching the sentinel on disconnect", () => {
    const h = harness();

    listOver(h).disconnect();

    expect(FakeObserver.latest!.watching).toBe(false);
  });

  it("is shown while next_cursor is non-null, even though the total is already painted", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a", "b"), 2, { ts: 300, id: "second-row" }));

    await listOver(h).load();

    expect(h.sentinelVisible()).toBe(true);
    expect(h.countText()).toBe("2 transcripts");
  });

  it("is hidden when next_cursor is null even though rows are still missing", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 99, null));

    await listOver(h).load();

    expect(h.sentinelVisible()).toBe(false);
  });
});

describe("createHistoryList — one request at a time decides the rows and the cursor", () => {
  it("names the version skew when the backend omits next_cursor entirely", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));

    await listOver(h).load();

    expect(h.countText()).toBe("History needs the latest backend — please update JustSay.");
  });

  it("keeps the version-skew warning painted when a row is deleted afterwards", async () => {
    const h = harness();
    apiMock.getHistory.mockResolvedValueOnce(pageOf(entries("a", "b"), 2, null));
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));

    const list = listOver(h);
    await list.load();
    await list.load();
    expect(h.countText()).toBe(sidecarTooOldText("History"));

    list.entryRemoved();

    expect(h.countText()).toBe(sidecarTooOldText("History"));
  });

  it("leaves the painted rows alone when a 200 carries a cursor but no page", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a", "b"), 2, null));
    const list = listOver(h);
    await list.load();
    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.countText()).toBe("2 transcripts");

    vi.stubGlobal("fetch", async () => ({
      ok: true,
      status: 200,
      json: async () => ({ next_cursor: null }),
    }));
    apiMock.getHistory.mockImplementation((limit: number, cursor: HistoryCursor | null) =>
      unmockedApi.getHistory(limit, cursor)
    );

    await list.load();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.countText()).toBe("Failed to load");
  });

  it("leaves the painted rows alone when a well-formed page carries a malformed entry", async () => {
    const h = harness();
    queueResponses(
      pageOf(entries("a", "b"), 2, null),
      pageOf([{ id: "c" } as unknown as HistoryEntry], 1, null)
    );
    const list = listOver(h, 2, () => false, createRowRequiringText);
    await list.load();
    expect(h.paintedIds()).toEqual(["a", "b"]);

    await list.load();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.countText()).toBe("Failed to load");
  });

  it("leaves the appended rows alone when a second page carries a malformed entry", async () => {
    const h = harness();
    queueResponses(
      pageOf(entries("a", "b"), 4, { ts: 1, id: "b" }),
      pageOf([buildEntry("c"), { id: "d" } as unknown as HistoryEntry], 4, null)
    );
    const list = listOver(h, 2, () => false, createRowRequiringText);
    await list.load();

    cross();
    await vi.waitFor(() => expect(h.countText()).toBe("Failed to load"));

    expect(h.paintedIds()).toEqual(["a", "b"]);
  });

  it("keeps the ordinary failure text for a failure that is not version skew", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValue(new Error("503 store busy"));

    await listOver(h).load();

    expect(h.countText()).toBe("Failed to load");
  });

  it("claimRows() holds paging and leaves the sentinel alone until the lane releases it", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    queueResponses(pageOf(entries("a", "b"), 4, cursor), pageOf(entries("c"), 4, null));

    const list = listOver(h);
    await list.load();
    const claim = list.claimRows();

    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
    expect(h.sentinelVisible()).toBe(true);

    claim.release();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    expect(sentCursors()).toEqual([null, cursor]);
  });

  it("a claim taken after teardown writes nothing", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a", "b"), 2, null));
    let destroyed = false;

    const list = listOver(h, 2, () => destroyed);
    await list.load();

    destroyed = true;
    const claim = list.claimRows();
    claim.renderCount("late");
    claim.replaceRows([]);

    expect(h.countText()).toBe("2 transcripts");
    expect(h.paintedIds()).toEqual(["a", "b"]);
  });

  it("only the current claim ends the hold, so a superseded append cannot start paging again", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 2 ? append.promise : pageOf(entries("a", "b"), 4, cursor);
    });

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));

    const claim = list.claimRows();
    append.release(pageOf(entries("c"), 4, null));
    await flush();

    expect(claim.isCurrent()).toBe(true);
    expect(h.paintedIds()).toEqual(["a", "b"]);
    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(2);

    claim.release();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));
  });

  it("clears the version-skew warning once a page comes back carrying a cursor", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValueOnce(new SidecarTooOldError("no next_cursor"));
    apiMock.getHistory.mockResolvedValue(pageOf(entries("a", "b"), 2, null));

    const list = listOver(h);
    await list.load();
    expect(h.countText()).toBe("History needs the latest backend — please update JustSay.");

    await list.load();

    expect(h.countText()).toBe("2 transcripts");

    list.entryRemoved();

    expect(h.countText()).toBe("1 transcript");
  });

  it("ignores a second crossing while the first page is still outstanding", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const second = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 1 ? pageOf(entries("a", "b"), 4, cursor) : second.promise;
    });

    const list = listOver(h);
    await list.load();
    cross();
    cross();
    second.release(pageOf(entries("c", "d"), 4, null));
    await flush();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(2);
    expect(h.paintedIds()).toEqual(["a", "b", "c", "d"]);
  });

  it("lets a reload supersede an append that is still in flight", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      if (calls === 2) return append.promise;
      return pageOf(entries("a", "b"), 4, cursor);
    });

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await list.load();
    append.release(pageOf(entries("c", "d"), 4, null));
    await flush();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.sentinelVisible()).toBe(true);

    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(4));
    expect(sentCursors()).toEqual([null, cursor, null, cursor]);
  });

  it("keeps paging held while the reload that superseded an append is outstanding", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    const reload = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      if (calls === 2) return append.promise;
      if (calls === 3) return reload.promise;
      return pageOf(entries("a", "b"), 4, cursor);
    });

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    void list.load();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));
    append.release(pageOf(entries("c"), 4, null));
    await flush();

    cross();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(3);
  });

  it("leaves the rows, the cursor and the sentinel untouched when a reload fails", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      if (calls === 2) throw new Error("503 store busy");
      if (calls === 3) return pageOf(entries("c", "d"), 4, null);
      return pageOf(entries("a", "b"), 4, cursor);
    });

    const list = listOver(h);
    await list.load();
    await list.load();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.sentinelVisible()).toBe(true);

    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));

    expect(sentCursors()).toEqual([null, null, cursor]);
    await vi.waitFor(() => expect(h.paintedIds()).toEqual(["a", "b", "c", "d"]));
  });
});

describe("createHistoryList — rows newer than the newest one painted", () => {
  it("asks after the newest row of the first page and puts what arrives on top, newest first", async () => {
    const h = harness();
    const store = entries("b", "a");
    queueResponses(pageOf([...store], 2, null));
    apiMock.getNewerHistory.mockImplementation(newerByCursor(store));

    const list = listOver(h, 5);
    await list.load();
    store.unshift(buildEntry("d"), buildEntry("c"));
    await list.loadNewer();

    expect(apiMock.getNewerHistory.mock.calls[0]).toEqual([5, positionOf(buildEntry("b"))]);
    expect(h.paintedIds()).toEqual(["d", "c", "b", "a"]);
    expect(h.countText()).toBe("4 transcripts");
  });

  it("never paints a row twice however often it asks", async () => {
    const h = harness();
    const store = entries("a");
    queueResponses(pageOf([...store], 1, null));
    apiMock.getNewerHistory.mockImplementation(newerByCursor(store));

    const list = listOver(h, 5);
    await list.load();
    store.unshift(buildEntry("b"));
    await list.loadNewer();
    await list.loadNewer();
    store.unshift(buildEntry("c"));
    await list.loadNewer();
    await list.loadNewer();

    expect(h.paintedIds()).toEqual(["c", "b", "a"]);
  });

  it("keeps asking while a newer page comes back full", async () => {
    const h = harness();
    const store = entries("a");
    queueResponses(pageOf([...store], 1, null));
    apiMock.getNewerHistory.mockImplementation(newerByCursor(store));

    const list = listOver(h, 2);
    await list.load();
    store.unshift(...entries("f", "e", "d", "c", "b"));
    await list.loadNewer();

    expect(apiMock.getNewerHistory).toHaveBeenCalledTimes(3);
    expect(h.paintedIds()).toEqual(["f", "e", "d", "c", "b", "a"]);
  });

  it("hands the newer page's days to the rows, so a grown day's header refreshes", async () => {
    const h = harness();
    const grown: HistoryDay = { date: "2026-08-01", recordings: 2, words: 8 };
    queueResponses(pageOf(entries("a"), 1, null));
    apiMock.getNewerHistory.mockResolvedValue(pageOf(entries("b"), 2, null, { days: [grown] }));

    const list = listOver(h, 5);
    await list.load();
    await list.loadNewer();

    expect(h.placedDays[h.placedDays.length - 1]).toEqual([grown]);
  });

  it("skips its turn while a page is being read", async () => {
    const h = harness();
    const second = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 1 ? pageOf(entries("a", "b"), 4, { ts: 1, id: "b" }) : second.promise;
    });

    const list = listOver(h);
    await list.load();
    cross();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await list.loadNewer();

    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();
  });

  it("skips its turn while another lane's rows are on screen", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 1, null));

    const list = listOver(h);
    await list.load();
    const claim = list.claimRows();
    claim.replaceRows([]);
    claim.release();
    await list.loadNewer();

    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();
  });

  it("asks for the first page again while the history it painted was empty", async () => {
    const h = harness();
    queueResponses(pageOf([], 0, null), pageOf(entries("a"), 1, null));

    const list = listOver(h);
    await list.load();
    await list.loadNewer();

    expect(sentCursors()).toEqual([null, null]);
    expect(apiMock.getNewerHistory).not.toHaveBeenCalled();
    expect(h.paintedIds()).toEqual(["a"]);
  });

  it("paints nothing and keeps the count when the read fails", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 1, null));
    apiMock.getNewerHistory.mockRejectedValue(new Error("503 store busy"));

    const list = listOver(h);
    await list.load();
    await list.loadNewer();

    expect(h.paintedIds()).toEqual(["a"]);
    expect(h.countText()).toBe("1 transcript");
  });

  it("drops a newer page that a reload superseded", async () => {
    const h = harness();
    const newer = deferredPage();
    queueResponses(pageOf(entries("a"), 1, null), pageOf(entries("a"), 1, null));
    apiMock.getNewerHistory.mockReturnValue(newer.promise);

    const list = listOver(h);
    await list.load();
    const poll = list.loadNewer();
    await list.load();
    newer.release(pageOf(entries("b"), 2, null));
    await poll;

    expect(h.paintedIds()).toEqual(["a"]);
  });
});

describe("createHistoryList — the count is the stored total, not the painted rows", () => {
  it("renders the total the response carried, not how many rows arrived", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a", "b"), 57, { ts: 300, id: "second-row" }));

    await listOver(h).load();

    expect(h.countText()).toBe("57 transcripts");
  });

  it("entryRemoved() decrements the displayed number and asks the backend for nothing", async () => {
    const h = harness();
    queueResponses(pageOf(entries("a"), 57, null));

    const list = listOver(h);
    await list.load();
    list.entryRemoved();

    expect(h.countText()).toBe("56 transcripts");
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
  });
});
