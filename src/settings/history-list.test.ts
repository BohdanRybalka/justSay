// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryCursor, HistoryPageResponse } from "../api";
import { buildEntry } from "./history-page-stub.test-helper";

const confirmMock = vi.fn();

vi.mock("@tauri-apps/plugin-dialog", () => ({
  confirm: confirmMock,
}));

const apiMock = {
  getHistory: vi.fn(),
  clearHistory: vi.fn(),
};

/**
 * Only `api` is replaced. Everything else in the module -- `SidecarTooOldError`
 * above all -- stays the real export, so the `instanceof` branch under test is
 * tied to the class `api.getHistory` actually throws. A stand-in class of the
 * same name passes whatever the module does, including throwing a plain
 * `Error`.
 */
vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

const { SidecarTooOldError } = await import("../api");
const { createHistoryList, formatEntryCount, sidecarTooOldText } = await import("./history-list");

const TRANSCRIPTS = { singular: "transcript", plural: "transcripts" };
const ENTRIES = { singular: "entry", plural: "entries" };

describe("formatEntryCount — the two tabs keep their own wording", () => {
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
  elements: {
    count: HTMLElement;
    rows: HTMLElement;
    loadMoreWrapper: HTMLElement;
    loadMoreButton: HTMLButtonElement;
    clearButton: HTMLButtonElement;
  };
  paintedIds: () => string[];
  loadMoreVisible: () => boolean;
  countText: () => string;
}

function harness(): Harness {
  const count = document.createElement("div");
  const rows = document.createElement("div");
  const loadMoreWrapper = document.createElement("div");
  const loadMoreButton = document.createElement("button");
  const clearButton = document.createElement("button");
  return {
    elements: { count, rows, loadMoreWrapper, loadMoreButton, clearButton },
    paintedIds: () => Array.from(rows.children).map((el) => el.textContent!),
    loadMoreVisible: () => loadMoreWrapper.style.display === "block",
    countText: () => count.textContent!,
  };
}

function listOver(h: Harness, pageSize = 2, isDestroyed: () => boolean = () => false) {
  return createHistoryList({
    pageSize,
    noun: TRANSCRIPTS,
    elements: h.elements,
    createRow: (entry) => {
      const row = document.createElement("div");
      row.textContent = entry.id;
      return row;
    },
    renderEmptyState: () => {},
    isDestroyed,
  });
}

/**
 * A backend the test drives by hand rather than by reimplementing the page
 * arithmetic. Each queued response is handed out in order; what is asserted is
 * the cursor the list *sent*, which a stub that computed its own answer would
 * hide.
 */
function queueResponses(...responses: HistoryPageResponse[]): void {
  let index = 0;
  apiMock.getHistory.mockImplementation(async () => responses[index++]);
}

function sentCursors(): (HistoryCursor | null)[] {
  return apiMock.getHistory.mock.calls.map((call) => call[1] as HistoryCursor | null);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("createHistoryList — the client echoes cursors and never builds one", () => {
  it("sends no cursor for the first page", async () => {
    const h = harness();
    queueResponses({ entries: [buildEntry("a")], total: 1, next_cursor: null });

    await listOver(h).load();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
    expect(apiMock.getHistory.mock.calls[0]).toEqual([2, null]);
  });

  it("sends the previous response's next_cursor back verbatim on Load more", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 1_700_000_000_042, id: "ff00ff00ff00" };
    queueResponses(
      { entries: [buildEntry("a"), buildEntry("b")], total: 3, next_cursor: cursor },
      { entries: [buildEntry("c")], total: 3, next_cursor: null }
    );

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => {
      expect(apiMock.getHistory).toHaveBeenCalledTimes(2);
    });

    expect(sentCursors()).toEqual([null, cursor]);
    expect(h.paintedIds()).toEqual(["a", "b", "c"]);
  });

  it("keeps paging from the newest cursor rather than the first one", async () => {
    const h = harness();
    const first: HistoryCursor = { ts: 300, id: "second-row" };
    const second: HistoryCursor = { ts: 100, id: "fourth-row" };
    queueResponses(
      { entries: [buildEntry("a"), buildEntry("b")], total: 5, next_cursor: first },
      { entries: [buildEntry("c"), buildEntry("d")], total: 5, next_cursor: second },
      { entries: [buildEntry("e")], total: 5, next_cursor: null }
    );

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));

    expect(sentCursors()).toEqual([null, first, second]);
    expect(h.paintedIds()).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("load() asks from the newest row again without discarding the stored cursor", async () => {
    const h = harness();
    const first: HistoryCursor = { ts: 300, id: "second-row" };
    const second: HistoryCursor = { ts: 100, id: "fourth-row" };
    queueResponses(
      { entries: [buildEntry("a"), buildEntry("b")], total: 5, next_cursor: first },
      { entries: [buildEntry("c"), buildEntry("d")], total: 5, next_cursor: second },
      { entries: [buildEntry("a"), buildEntry("b")], total: 5, next_cursor: first }
    );

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await list.load();

    expect(sentCursors()).toEqual([null, first, null]);
    expect(h.paintedIds()).toEqual(["a", "b"]);
  });

  it("clearAll() drops the cursor, so nothing can page into the deleted history", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 5 });
    queueResponses({ entries: [buildEntry("a"), buildEntry("b")], total: 5, next_cursor: cursor });

    const list = listOver(h);
    await list.load();
    h.elements.clearButton.click();
    await vi.waitFor(() => expect(apiMock.clearHistory).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(h.countText()).toBe("0 transcripts"));

    h.elements.loadMoreButton.click();
    await flush();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
    expect(sentCursors()).toEqual([null]);
  });

  it("refuses an append while the stored cursor is null instead of re-asking for page one", async () => {
    const h = harness();
    queueResponses({ entries: [buildEntry("a")], total: 1, next_cursor: null });

    await listOver(h).load();
    h.elements.loadMoreButton.click();
    await flush();

    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
    expect(sentCursors()).toEqual([null]);
  });

  it("leaves the button as it found it when it refuses an append", async () => {
    const h = harness();
    queueResponses({ entries: [buildEntry("a")], total: 1, next_cursor: null });

    await listOver(h).load();
    expect(h.elements.loadMoreButton.disabled).toBe(false);

    h.elements.loadMoreButton.click();
    await flush();

    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });
});

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

describe("createHistoryList — one request at a time decides the rows and the cursor", () => {
  it("names the version skew when the backend omits next_cursor entirely", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));

    await listOver(h).load();

    expect(h.countText()).toBe("History needs the latest backend — please update JustSay.");
  });

  it("keeps the version-skew warning painted when a row is deleted afterwards", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));

    const list = listOver(h);
    await list.load();
    list.entryRemoved();

    expect(h.countText()).toBe("History needs the latest backend — please update JustSay.");
  });

  it("keeps the ordinary failure text for a failure that is not version skew", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValue(new Error("503 store busy"));

    await listOver(h).load();

    expect(h.countText()).toBe("Failed to load");
  });

  it("claimRows() disables Load more and leaves the wrapper alone, so an unrepainted list stays pageable", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    queueResponses({ entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor });

    const list = listOver(h);
    await list.load();
    expect(h.loadMoreVisible()).toBe(true);

    const claim = list.claimRows();

    expect(h.loadMoreVisible()).toBe(true);
    expect(h.elements.loadMoreButton.disabled).toBe(true);

    h.elements.loadMoreButton.click();
    await flush();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);

    claim.release();

    expect(h.loadMoreVisible()).toBe(true);
    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });

  it("claimRows() touches nothing once the tab is gone", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    queueResponses({ entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor });
    let destroyed = false;

    const list = listOver(h, 2, () => destroyed);
    await list.load();
    expect(h.elements.loadMoreButton.disabled).toBe(false);

    destroyed = true;
    list.claimRows();

    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });

  it("only the current claim hands Load more back, so a superseded append cannot re-enable it", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 2
        ? append.promise
        : { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor };
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    expect(h.elements.loadMoreButton.disabled).toBe(true);

    const claim = list.claimRows();

    append.release({ entries: [buildEntry("c")], total: 4, next_cursor: null });
    await flush();

    expect(claim.isCurrent()).toBe(true);
    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.elements.loadMoreButton.disabled).toBe(true);

    claim.release();

    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });

  it("clears the version-skew warning once a page comes back carrying a cursor", async () => {
    const h = harness();
    apiMock.getHistory.mockRejectedValueOnce(new SidecarTooOldError("no next_cursor"));
    apiMock.getHistory.mockResolvedValue({
      entries: [buildEntry("a"), buildEntry("b")],
      total: 2,
      next_cursor: null,
    });

    const list = listOver(h);
    await list.load();
    expect(h.countText()).toBe("History needs the latest backend — please update JustSay.");

    await list.load();

    expect(h.countText()).toBe("2 transcripts");

    list.entryRemoved();

    expect(h.countText()).toBe("1 transcript");
  });

  it("keeps the version-skew warning through a Clear All, because the backend is still the old one", async () => {
    const h = harness();
    apiMock.getHistory.mockResolvedValueOnce({
      entries: [buildEntry("a")],
      total: 5,
      next_cursor: null,
    });
    apiMock.getHistory.mockRejectedValue(new SidecarTooOldError("no next_cursor"));
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 5 });

    const list = listOver(h);
    await list.load();
    await list.load();
    expect(h.countText()).toBe(sidecarTooOldText("History"));

    h.elements.clearButton.click();
    await vi.waitFor(() => expect(apiMock.clearHistory).toHaveBeenCalledTimes(1));
    await flush();

    expect(h.countText()).toBe(sidecarTooOldText("History"));
    expect(h.loadMoreVisible()).toBe(false);
    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });

  it("clearAll() paints the empty count and hides Load more through the claim it took", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    queueResponses({ entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor });
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue(undefined);

    await listOver(h).load();
    h.elements.clearButton.click();
    await vi.waitFor(() => expect(apiMock.clearHistory).toHaveBeenCalledTimes(1));

    await vi.waitFor(() => expect(h.countText()).toBe("0 transcripts"));
    expect(h.loadMoreVisible()).toBe(false);
    expect(h.elements.loadMoreButton.disabled).toBe(false);
  });

  it("ignores a second Load more click while the first is still outstanding", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const second = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 1
        ? { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor }
        : second.promise;
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    h.elements.loadMoreButton.click();
    second.release({
      entries: [buildEntry("c"), buildEntry("d")],
      total: 4,
      next_cursor: null,
    });
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
      return { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor };
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    await list.load();
    append.release({ entries: [buildEntry("c"), buildEntry("d")], total: 4, next_cursor: null });
    await flush();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.loadMoreVisible()).toBe(true);

    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(4));
    expect(sentCursors()).toEqual([null, cursor, null, cursor]);
  });

  it("keeps Load more refused while the reload that superseded an append is outstanding", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    const reload = deferredPage();
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      if (calls === 2) return append.promise;
      if (calls === 3) return reload.promise;
      return { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor };
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    void list.load();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));
    append.release({ entries: [buildEntry("c")], total: 4, next_cursor: null });
    await flush();

    expect(h.elements.loadMoreButton.disabled).toBe(true);
    h.elements.loadMoreButton.click();
    expect(apiMock.getHistory).toHaveBeenCalledTimes(3);
  });

  it("leaves the rows, the cursor and Load more untouched when a reload fails", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      if (calls === 2) throw new Error("503 store busy");
      if (calls === 3) {
        return { entries: [buildEntry("c"), buildEntry("d")], total: 4, next_cursor: null };
      }
      return { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor };
    });

    const list = listOver(h);
    await list.load();
    await list.load();

    expect(h.paintedIds()).toEqual(["a", "b"]);
    expect(h.loadMoreVisible()).toBe(true);

    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(3));

    expect(sentCursors()).toEqual([null, null, cursor]);
    expect(h.paintedIds()).toEqual(["a", "b", "c", "d"]);
  });

  it("leaves Load more clickable after clearAll() superseded an outstanding append", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 4 });
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 2
        ? append.promise
        : { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor };
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    h.elements.clearButton.click();
    await vi.waitFor(() => expect(h.countText()).toBe("0 transcripts"));
    append.release({ entries: [buildEntry("c")], total: 4, next_cursor: null });
    await flush();

    expect(h.elements.loadMoreButton.disabled).toBe(false);
    expect(sentCursors()).toEqual([null, cursor]);
  });

  it("lets clearAll() supersede an append that is still in flight", async () => {
    const h = harness();
    const cursor: HistoryCursor = { ts: 300, id: "second-row" };
    const append = deferredPage();
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 4 });
    let calls = 0;
    apiMock.getHistory.mockImplementation(async () => {
      calls += 1;
      return calls === 1
        ? { entries: [buildEntry("a"), buildEntry("b")], total: 4, next_cursor: cursor }
        : append.promise;
    });

    const list = listOver(h);
    await list.load();
    h.elements.loadMoreButton.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    h.elements.clearButton.click();
    await vi.waitFor(() => expect(h.countText()).toBe("0 transcripts"));
    append.release({ entries: [buildEntry("c"), buildEntry("d")], total: 4, next_cursor: null });
    await flush();

    expect(h.paintedIds()).toEqual([]);
    expect(h.countText()).toBe("0 transcripts");
    expect(h.loadMoreVisible()).toBe(false);
  });
});

describe("createHistoryList — Load more tracks next_cursor, not the total", () => {
  it("shows the control when next_cursor is non-null", async () => {
    const h = harness();
    queueResponses({
      entries: [buildEntry("a"), buildEntry("b")],
      total: 2,
      next_cursor: { ts: 300, id: "second-row" },
    });

    await listOver(h).load();

    expect(h.loadMoreVisible()).toBe(true);
  });

  it("hides the control when next_cursor is null even though rows are still missing", async () => {
    const h = harness();
    queueResponses({ entries: [buildEntry("a")], total: 99, next_cursor: null });

    await listOver(h).load();

    expect(h.loadMoreVisible()).toBe(false);
  });

  it("shows the control when next_cursor is non-null even though the total is already painted", async () => {
    const h = harness();
    queueResponses({
      entries: [buildEntry("a"), buildEntry("b")],
      total: 2,
      next_cursor: { ts: 300, id: "second-row" },
    });

    await listOver(h).load();

    expect(h.loadMoreVisible()).toBe(true);
    expect(h.countText()).toBe("2 transcripts");
  });
});

describe("createHistoryList — the count is the stored total, not the painted rows", () => {
  it("renders the total the response carried, not how many rows arrived", async () => {
    const h = harness();
    queueResponses({
      entries: [buildEntry("a"), buildEntry("b")],
      total: 57,
      next_cursor: { ts: 300, id: "second-row" },
    });

    await listOver(h).load();

    expect(h.countText()).toBe("57 transcripts");
  });

  it("entryRemoved() decrements the displayed number and asks the backend for nothing", async () => {
    const h = harness();
    queueResponses({ entries: [buildEntry("a")], total: 57, next_cursor: null });

    const list = listOver(h);
    await list.load();
    list.entryRemoved();

    expect(h.countText()).toBe("56 transcripts");
    expect(apiMock.getHistory).toHaveBeenCalledTimes(1);
  });
});
