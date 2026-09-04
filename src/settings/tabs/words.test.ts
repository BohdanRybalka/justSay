// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryStats, TopWordsResponse } from "../../api";

const { apiMock } = vi.hoisted(() => ({
  apiMock: { historyStats: vi.fn(), wordsTop: vi.fn() },
}));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

import { renderWords } from "./words";

function buildStats(overrides: Partial<HistoryStats> = {}): HistoryStats {
  return {
    total_entries: 1,
    total_words: 100,
    today_words: 10,
    week_words: 50,
    total_audio_seconds: 60,
    by_language: {},
    by_model: {},
    ...overrides,
  } as HistoryStats;
}

const noTopWords: TopWordsResponse = { items: [], scanned: 0 };

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  apiMock.wordsTop.mockResolvedValue(noTopWords);
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
});

describe("the Words tab's 5 s poll", () => {
  it("paints the newest answer, not whichever overlapping probe finished last", async () => {
    const settle: Array<(stats: HistoryStats) => void> = [];
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_words: 100 }));
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => settle.push(resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const destroy = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(5000);
    expect(settle).toHaveLength(2);

    settle[1](buildStats({ total_words: 222 }));
    await vi.advanceTimersByTimeAsync(0);
    settle[0](buildStats({ total_words: 111 }));
    await vi.advanceTimersByTimeAsync(0);

    expect(document.getElementById("words-stat-lifetime")!.textContent).toBe(
      (222).toLocaleString("uk-UA"),
    );

    destroy();
    container.remove();
  });
});

describe("the whole-page read the empty-to-non-empty transition triggers", () => {
  it("holds the poll off while it runs, so its own answer is the one that lands", async () => {
    const settle: Array<(stats: HistoryStats) => void> = [];
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 0 }));
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 5, total_words: 55 }));
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => settle.push(resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const destroy = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("No transcriptions yet");

    await vi.advanceTimersByTimeAsync(5000);
    expect(settle).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(5000);
    expect(settle).toHaveLength(1);

    settle[0](buildStats({ total_entries: 5, total_words: 55 }));
    await vi.advanceTimersByTimeAsync(0);

    expect(container.textContent).not.toContain("No transcriptions yet");
    expect(document.getElementById("words-stat-lifetime")!.textContent).toBe(
      (55).toLocaleString("uk-UA"),
    );

    destroy();
    container.remove();
  });
});
