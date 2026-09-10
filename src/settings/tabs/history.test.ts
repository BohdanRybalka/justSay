// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryEntry, HistoryPageResponse } from "../../api";
import { buildEntry, pagesByCursor } from "../history-page-stub.test-helper";

const confirmMock = vi.fn();

vi.mock("@tauri-apps/plugin-dialog", () => ({
  confirm: confirmMock,
}));

const apiMock = {
  getHistory: vi.fn(),
  searchHistory: vi.fn(),
  clearHistory: vi.fn(),
  deleteHistoryEntry: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
}));

const { renderHistory } = await import("./history");

async function renderWith(total: number): Promise<HTMLElement> {
  const entries = Array.from({ length: total }, (_, index) => buildEntry(String(index + 1)));
  apiMock.getHistory.mockResolvedValue({ entries, total, next_cursor: null });
  const container = document.createElement("div");
  renderHistory(container);
  await vi.waitFor(() => {
    const plural = total !== 1 ? "s" : "";
    expect(container.querySelector("#history-count")!.textContent).toBe(
      `${total} transcript${plural}`
    );
  });
  return container;
}

async function renderPaged(total: number): Promise<HTMLElement> {
  const all = Array.from({ length: total }, (_, index) => buildEntry(String(index + 1)));
  apiMock.getHistory.mockImplementation(pagesByCursor(all));
  const container = document.createElement("div");
  renderHistory(container);
  await vi.waitFor(() => {
    expect(container.querySelector("#history-count")!.textContent).not.toBe("Loading...");
  });
  return container;
}

function clearButton(container: HTMLElement): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>("#btn-clear-history")!;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("renderHistory — paging over the history endpoint", () => {
  it("asks for 30 transcripts on the first paint", async () => {
    const container = await renderPaged(40);

    expect(apiMock.getHistory.mock.calls[0]).toEqual([30, null]);
    expect(container.querySelectorAll(".history-entry")).toHaveLength(30);
    expect(container.querySelector<HTMLElement>("#history-load-more")!.style.display).toBe("block");
  });

  it("one Load more click brings in the rest and hides the wrapper", async () => {
    const container = await renderPaged(40);

    container.querySelector<HTMLButtonElement>("#btn-load-more")!.click();

    await vi.waitFor(() => {
      expect(container.querySelectorAll(".history-entry")).toHaveLength(40);
    });
    expect(apiMock.getHistory.mock.calls[1][0]).toBe(30);
    expect(apiMock.getHistory.mock.calls[1][1]).toEqual({ ts: 30, id: "30" });
    expect(container.querySelector<HTMLElement>("#history-load-more")!.style.display).toBe("none");
  });
});

describe("renderHistory — the count names transcripts", () => {
  it("reads '1 transcript' for a single entry", async () => {
    const container = await renderWith(1);
    expect(container.querySelector("#history-count")!.textContent).toBe("1 transcript");
  });

  it("reads '2 transcripts' for two entries", async () => {
    const container = await renderWith(2);
    expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
  });
});

describe("renderHistory — teardown", () => {
  it("a response arriving after teardown writes nothing", async () => {
    let release: (value: HistoryPageResponse) => void = () => {};
    apiMock.getHistory.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      })
    );
    const container = document.createElement("div");
    const teardown = renderHistory(container);

    const before = container.querySelector("#history-count")!.textContent;
    teardown();
    release({ entries: [buildEntry("1")], total: 1, next_cursor: null });
    await Promise.resolve();
    await Promise.resolve();

    expect(container.querySelector("#history-count")!.textContent).toBe(before);
    expect(container.querySelectorAll(".history-entry")).toHaveLength(0);
  });

  it("a search resolving after teardown writes nothing", async () => {
    let release: (value: { entries: HistoryEntry[]; total: number }) => void = () => {};
    apiMock.searchHistory.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      })
    );
    const entries = [buildEntry("1"), buildEntry("2")];
    apiMock.getHistory.mockResolvedValue({ entries, total: 2, next_cursor: null });
    const container = document.createElement("div");
    const teardown = renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    });

    const search = container.querySelector<HTMLInputElement>("#history-search")!;
    search.value = "hello";
    search.dispatchEvent(new Event("input"));
    await vi.waitFor(() => {
      expect(apiMock.searchHistory).toHaveBeenCalledTimes(1);
    });

    const countBefore = container.querySelector("#history-count")!.textContent;
    teardown();
    release({ entries: [buildEntry("9")], total: 1 });
    await Promise.resolve();
    await Promise.resolve();

    expect(container.querySelector("#history-count")!.textContent).toBe(countBefore);
    expect(container.querySelectorAll(".history-entry")).toHaveLength(2);
  });
});

/**
 * The two lanes that can paint History's rows, driven against each other.
 *
 * `renderHistory` paints once on mount, so the reload under test is the one the
 * search box issues when it is emptied; the query typed afterwards is what takes
 * the rows over while that reload is still outstanding. Both answers are held
 * open by hand so the order they arrive in is the test's choice rather than the
 * scheduler's.
 */
function deferred<T>(): { promise: Promise<T>; release: (value: T) => void } {
  let release: (value: T) => void = () => {};
  const promise = new Promise<T>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

function typeQuery(container: HTMLElement, value: string): void {
  const search = container.querySelector<HTMLInputElement>("#history-search")!;
  search.value = value;
  search.dispatchEvent(new Event("input"));
}

async function deleteFirstRow(container: HTMLElement): Promise<void> {
  const entry = container.querySelector<HTMLElement>(".history-entry")!;
  entry.querySelector<HTMLButtonElement>('[data-action="delete"]')!.click();
  await vi.waitFor(() => {
    expect(apiMock.deleteHistoryEntry).toHaveBeenCalledTimes(1);
  });
}

describe("renderHistory — a reload and a search cannot both own the rows", () => {
  it("a search that answers first keeps its matches when the reload arrives after it", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    const reload = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1 ? { entries, total: 2, next_cursor: null } : reload.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });

    const container = document.createElement("div");
    renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    });

    typeQuery(container, "");
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));

    search.release({ entries: [buildEntry("9")], total: 1 });
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("1 match");
    });

    reload.release({ entries, total: 2, next_cursor: { ts: 1, id: "1" } });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(container.querySelector("#history-count")!.textContent).toBe("1 match");
    expect(container.querySelectorAll(".history-entry")).toHaveLength(1);
    expect(container.querySelector<HTMLElement>("#history-load-more")!.style.display).toBe("none");

    await deleteFirstRow(container);
    expect(container.querySelector("#history-count")!.textContent).toBe("1 match");
  });

  it("a reload that takes the rows over drops the search answer that arrives after it", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    const reload = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1 ? { entries, total: 2, next_cursor: null } : reload.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    apiMock.deleteHistoryEntry.mockResolvedValue({ deleted: true });

    const container = document.createElement("div");
    renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    });

    typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    typeQuery(container, "");
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));

    reload.release({ entries, total: 2, next_cursor: null });
    await vi.waitFor(() => {
      expect(container.querySelectorAll(".history-entry")).toHaveLength(2);
    });

    search.release({ entries: [buildEntry("9")], total: 1 });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    expect(container.querySelectorAll(".history-entry")).toHaveLength(2);

    await deleteFirstRow(container);
    expect(container.querySelector("#history-count")!.textContent).toBe("1 transcript");
  });

  it("a search taking the rows over leaves Load more clickable rather than dead", async () => {
    const entries = Array.from({ length: 30 }, (_, index) => buildEntry(String(index + 1)));
    const append = deferred<HistoryPageResponse>();
    let pages = 0;
    apiMock.getHistory.mockImplementation(async () => {
      pages += 1;
      return pages === 1
        ? { entries, total: 60, next_cursor: { ts: 30, id: "30" } }
        : append.promise;
    });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);

    const container = document.createElement("div");
    renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("60 transcripts");
    });

    const loadMore = container.querySelector<HTMLButtonElement>("#btn-load-more")!;
    loadMore.click();
    await vi.waitFor(() => expect(apiMock.getHistory).toHaveBeenCalledTimes(2));
    expect(loadMore.disabled).toBe(true);

    typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));

    expect(loadMore.disabled).toBe(false);

    search.release({ entries: [], total: 0 });
    append.release({ entries: [buildEntry("31")], total: 60, next_cursor: null });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(loadMore.disabled).toBe(false);
  });
});

describe("renderHistory — the search hint belongs to the lane that put it up", () => {
  it("clears Searching... when Clear All takes the rows over before the search answers", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    apiMock.getHistory.mockResolvedValue({ entries, total: 2, next_cursor: null });
    const search = deferred<{ entries: HistoryEntry[]; total: number }>();
    apiMock.searchHistory.mockReturnValue(search.promise);
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 2 });

    const container = document.createElement("div");
    renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    });

    typeQuery(container, "hello");
    await vi.waitFor(() => expect(apiMock.searchHistory).toHaveBeenCalledTimes(1));
    expect(container.querySelector("#history-search-hint")!.textContent).toBe("Searching...");

    clearButton(container).click();
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("0 transcripts");
    });

    search.release({ entries: [buildEntry("9")], total: 1 });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(container.querySelector("#history-search-hint")!.textContent).toBe("");
    expect(container.querySelector("#history-count")!.textContent).toBe("0 transcripts");
    expect(container.querySelectorAll(".history-entry")).toHaveLength(0);
  });

  it("keeps a failing search's own message, which the lane still owns", async () => {
    const entries = [buildEntry("1"), buildEntry("2")];
    apiMock.getHistory.mockResolvedValue({ entries, total: 2, next_cursor: null });
    apiMock.searchHistory.mockRejectedValue(new Error("503 store busy"));

    const container = document.createElement("div");
    renderHistory(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("2 transcripts");
    });

    typeQuery(container, "hello");
    await vi.waitFor(() => {
      expect(container.querySelector("#history-search-hint")!.textContent).toBe("503 store busy");
    });

    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(container.querySelector("#history-search-hint")!.textContent).toBe("503 store busy");
  });
});

describe("renderHistory — Clear All asks before deleting everything", () => {
  it("cancelling the dialog leaves every transcript in place", async () => {
    confirmMock.mockResolvedValue(false);
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(confirmMock).toHaveBeenCalledTimes(1);
    });
    expect(apiMock.clearHistory).not.toHaveBeenCalled();
    expect(container.querySelectorAll(".history-entry")).toHaveLength(2);
  });

  it("cancelling re-enables the button instead of leaving it dead", async () => {
    confirmMock.mockResolvedValue(false);
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(clearButton(container).disabled).toBe(false);
    });
    expect(clearButton(container).textContent).toBe("Clear All");
  });

  it("the dialog names how many transcripts go and that Metrics shares them", async () => {
    confirmMock.mockResolvedValue(false);
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(confirmMock).toHaveBeenCalledTimes(1);
    });
    const [message] = confirmMock.mock.calls[0];
    expect(message).toContain("2 transcripts");
    expect(message).toContain("Metrics");
  });

  it("confirming the dialog clears the list", async () => {
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 2 });
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(apiMock.clearHistory).toHaveBeenCalledTimes(1);
    });
    await vi.waitFor(() => {
      expect(container.querySelector("#history-count")!.textContent).toBe("0 transcripts");
    });
    expect(container.querySelectorAll(".history-entry")).toHaveLength(0);
  });
});
