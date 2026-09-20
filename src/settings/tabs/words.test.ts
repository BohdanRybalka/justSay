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
  document.body.innerHTML = "";
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
    const tab = renderWords(container);
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

    tab.destroy();
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
    const tab = renderWords(container);
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

    tab.destroy();
    container.remove();
  });
});

describe("the Words tab after the Settings window is dismissed", () => {
  async function mountOnStats() {
    apiMock.historyStats.mockResolvedValue(buildStats());
    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    return { tab, container };
  }

  it("issues no further stats read once the window is dismissed", async () => {
    const { tab, container } = await mountOnStats();

    tab.releaseResources!();
    const whileHidden = apiMock.historyStats.mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);

    expect(apiMock.historyStats.mock.calls.length).toBe(whileHidden);

    tab.destroy();
    container.remove();
  });

  it("reads once on the way back before any tick, then keeps the 5 s rhythm", async () => {
    const { tab, container } = await mountOnStats();

    tab.releaseResources!();
    const whileHidden = apiMock.historyStats.mock.calls.length;

    tab.resumeResources!();

    expect(
      apiMock.historyStats.mock.calls.length,
      "a returning user reads figures one request old, not one interval old",
    ).toBe(whileHidden + 1);

    await vi.advanceTimersByTimeAsync(5000);
    expect(apiMock.historyStats.mock.calls.length).toBe(whileHidden + 2);

    tab.destroy();
    container.remove();
  });

  it("starts no second interval when a resume lands on a poll already running", async () => {
    const { tab, container } = await mountOnStats();

    tab.resumeResources!();
    const afterResume = apiMock.historyStats.mock.calls.length;

    await vi.advanceTimersByTimeAsync(5000);
    expect(
      apiMock.historyStats.mock.calls.length,
      "a resume that overwrites the live handle reads twice per tick",
    ).toBe(afterResume + 1);

    tab.releaseResources!();
    await vi.advanceTimersByTimeAsync(30_000);

    expect(
      apiMock.historyStats.mock.calls.length,
      "and leaves an interval the release can no longer reach",
    ).toBe(afterResume + 1);

    tab.destroy();
    container.remove();
  });

  it("drops a stats read the dismissal caught in flight rather than chaining a page read", async () => {
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 0 }));
    let settleTheTick: (stats: HistoryStats) => void = () => {};
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => (settleTheTick = resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(5000);
    const inFlight = apiMock.historyStats.mock.calls.length;

    tab.releaseResources!();
    settleTheTick(buildStats({ total_entries: 5 }));
    await vi.advanceTimersByTimeAsync(0);

    expect(
      apiMock.historyStats.mock.calls.length,
      "an answer landing after the dismissal chains into the whole-page read: two more " +
        "requests and a repaint of a window nobody is looking at",
    ).toBe(inFlight);

    tab.destroy();
    container.remove();
  });

  it("repairs a first page read that failed when the window comes back", async () => {
    apiMock.historyStats.mockRejectedValueOnce(new Error("the backend went away"));

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("Failed to load");

    tab.releaseResources!();
    apiMock.historyStats.mockResolvedValue(buildStats({ total_words: 4321 }));
    tab.resumeResources!();
    await vi.advanceTimersByTimeAsync(0);

    expect(
      document.getElementById("words-stat-lifetime")?.textContent,
      "a tab whose first page read failed returns early on every later tick, so re-opening " +
        "the window is the only thing left that can repair it",
    ).toBe((4321).toLocaleString("uk-UA"));

    tab.destroy();
    container.remove();
  });
});
