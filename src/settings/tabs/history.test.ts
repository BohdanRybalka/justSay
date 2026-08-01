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
  apiMock.getHistory.mockResolvedValue({ entries, total });
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

function clearButton(container: HTMLElement): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>("#btn-clear-history")!;
}

beforeEach(() => {
  vi.clearAllMocks();
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
