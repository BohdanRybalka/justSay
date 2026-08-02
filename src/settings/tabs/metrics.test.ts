// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryEntry } from "../../api";

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

const { renderMetrics } = await import("./metrics");

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

function stubBackend(total: number): HistoryEntry[] {
  const all = Array.from({ length: total }, (_, index) => buildEntry(String(index + 1)));
  apiMock.getHistory.mockImplementation(async (limit: number, offset: number) => ({
    entries: all.slice(offset, offset + limit),
    total,
  }));
  return all;
}

async function renderWith(total: number): Promise<HTMLElement> {
  stubBackend(total);
  const container = document.createElement("div");
  renderMetrics(container);
  await vi.waitFor(() => {
    expect(container.querySelector("#metrics-count")!.textContent).not.toBe("Loading...");
  });
  return container;
}

function rowCount(container: HTMLElement): number {
  return container.querySelectorAll("#metrics-body tr").length;
}

function clearButton(container: HTMLElement): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>("#btn-clear-metrics")!;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("renderMetrics — paging over the history endpoint", () => {
  it("asks for 50 rows on the first paint", async () => {
    const container = await renderWith(60);

    expect(apiMock.getHistory.mock.calls[0]).toEqual([50, 0]);
    expect(rowCount(container)).toBe(50);
    expect(container.querySelector<HTMLElement>("#metrics-load-more")!.style.display).toBe("block");
  });

  it("one Load more click brings in the remaining rows and hides the button", async () => {
    const container = await renderWith(60);

    container.querySelector<HTMLButtonElement>("#btn-load-more-metrics")!.click();

    await vi.waitFor(() => {
      expect(rowCount(container)).toBe(60);
    });
    expect(apiMock.getHistory.mock.calls[1]).toEqual([50, 50]);
    expect(container.querySelector<HTMLElement>("#metrics-load-more")!.style.display).toBe("none");
  });

  it("an empty backend shows the empty state instead of the table", async () => {
    const container = await renderWith(0);

    expect(container.querySelector<HTMLElement>("#metrics-empty")!.style.display).toBe("block");
    expect(container.querySelector<HTMLElement>("#metrics-table-wrap")!.style.display).toBe("none");
    expect(container.querySelector("#metrics-count")!.textContent).toBe("0 entries");
  });
});

describe("renderMetrics — the count names entries, not transcripts", () => {
  it("reads '1 entry' for a single row", async () => {
    const container = await renderWith(1);
    expect(container.querySelector("#metrics-count")!.textContent).toBe("1 entry");
  });

  it("reads '2 entries' for two rows", async () => {
    const container = await renderWith(2);
    expect(container.querySelector("#metrics-count")!.textContent).toBe("2 entries");
  });
});

describe("renderMetrics — Clear All asks before deleting everything", () => {
  it("declining the dialog deletes nothing and re-enables the button", async () => {
    confirmMock.mockResolvedValue(false);
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(confirmMock).toHaveBeenCalledTimes(1);
    });
    expect(apiMock.clearHistory).not.toHaveBeenCalled();
    expect(rowCount(container)).toBe(2);
    await vi.waitFor(() => {
      expect(clearButton(container).disabled).toBe(false);
    });
    expect(clearButton(container).textContent).toBe("Clear All");
  });

  it("the dialog names how many entries go and that History shares them", async () => {
    confirmMock.mockResolvedValue(false);
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(confirmMock).toHaveBeenCalledTimes(1);
    });
    expect(confirmMock.mock.calls[0][0]).toBe(
      "Delete all 2 entries? History and Metrics share the same data — both tabs will be cleared."
    );
  });

  it("accepting the dialog empties the table and the count", async () => {
    confirmMock.mockResolvedValue(true);
    apiMock.clearHistory.mockResolvedValue({ deleted: 2 });
    const container = await renderWith(2);

    clearButton(container).click();

    await vi.waitFor(() => {
      expect(apiMock.clearHistory).toHaveBeenCalledTimes(1);
    });
    await vi.waitFor(() => {
      expect(container.querySelector("#metrics-count")!.textContent).toBe("0 entries");
    });
    expect(rowCount(container)).toBe(0);
    expect(container.querySelector<HTMLElement>("#metrics-empty")!.style.display).toBe("block");
    expect(container.querySelector<HTMLElement>("#metrics-table-wrap")!.style.display).toBe("none");
  });
});

describe("renderMetrics — teardown", () => {
  it("a response arriving after teardown writes nothing", async () => {
    let release: (value: { entries: HistoryEntry[]; total: number }) => void = () => {};
    apiMock.getHistory.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      })
    );
    const container = document.createElement("div");
    const teardown = renderMetrics(container);

    const before = container.querySelector("#metrics-count")!.textContent;
    teardown();
    release({ entries: [buildEntry("1")], total: 1 });
    await Promise.resolve();
    await Promise.resolve();

    expect(container.querySelector("#metrics-count")!.textContent).toBe(before);
    expect(rowCount(container)).toBe(0);
  });

  it("a Clear All confirmed after teardown still deletes but paints nothing", async () => {
    let approve: (value: boolean) => void = () => {};
    confirmMock.mockReturnValue(
      new Promise<boolean>((resolve) => {
        approve = resolve;
      })
    );
    apiMock.clearHistory.mockResolvedValue({ deleted: 2 });
    stubBackend(2);
    const container = document.createElement("div");
    const teardown = renderMetrics(container);
    await vi.waitFor(() => {
      expect(container.querySelector("#metrics-count")!.textContent).not.toBe("Loading...");
    });

    clearButton(container).click();
    await vi.waitFor(() => {
      expect(confirmMock).toHaveBeenCalledTimes(1);
    });

    const countBefore = container.querySelector("#metrics-count")!.textContent;
    teardown();
    approve(true);
    await vi.waitFor(() => {
      expect(apiMock.clearHistory).toHaveBeenCalledTimes(1);
    });

    expect(container.querySelector("#metrics-count")!.textContent).toBe(countBefore);
    expect(rowCount(container)).toBe(2);
    expect(clearButton(container).textContent).toBe("Clear All");
    expect(clearButton(container).disabled).toBe(true);
  });
});
