// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryCursor, HistoryEntry, HistoryPageResponse } from "../../api";

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

function buildEntry(id: string): HistoryEntry {
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
  /** Serves the page the cursor names, found by identity rather than by counting
   *  how many rows the stub has already handed out: a null cursor is the first
   *  page, and a cursor is the rows strictly after the entry it names. A stub that
   *  counts cannot tell a client that echoes `next_cursor` from one that sends
   *  `null` every time, and answers a reset cursor with page 2. */
  apiMock.getHistory.mockImplementation(async (limit: number, cursor: HistoryCursor | null) => {
    const start = cursor === null ? 0 : all.findIndex((entry) => entry.id === cursor.id) + 1;
    const entries = all.slice(start, start + limit);
    const last = entries[entries.length - 1];
    const end = start + entries.length;
    return {
      entries,
      total,
      next_cursor: last && end < all.length ? { ts: end, id: last.id } : null,
    };
  });
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
