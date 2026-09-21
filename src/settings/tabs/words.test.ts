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
      "a returning user is shown the repair in the same tick as the window, rather than " +
        "left on the error line until the next poll fires five seconds later",
    ).toBe((4321).toLocaleString("uk-UA"));

    tab.destroy();
    container.remove();
  });

  it("issues no follow-up request for a page read the dismissal caught in flight", async () => {
    let settlePage: (stats: HistoryStats) => void = () => {};
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => (settlePage = resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);

    tab.releaseResources!();
    settlePage(buildStats({ total_entries: 3 }));
    await vi.advanceTimersByTimeAsync(0);

    expect(
      apiMock.wordsTop,
      "the whole-page read is the one that costs a second request and a full repaint, " +
        "so the dismissal has to reach it and not only the tick",
    ).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Loading...");

    tab.destroy();
    container.remove();
  });

  it("lets no page read the dismissal disowned repaint over the one that replaced it", async () => {
    const settle: Array<(stats: HistoryStats) => void> = [];
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => settle.push(resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(settle).toHaveLength(1);

    tab.releaseResources!();
    tab.resumeResources!();
    expect(settle).toHaveLength(2);

    settle[1](buildStats({ total_words: 999 }));
    await vi.advanceTimersByTimeAsync(0);
    settle[0](buildStats({ total_words: 111 }));
    await vi.advanceTimersByTimeAsync(0);

    expect(
      document.getElementById("words-stat-lifetime")!.textContent,
      "the read the returning user is waiting on is the newer one, and the answer the " +
        "dismissal disowned must not paint the figures it read before the window closed",
    ).toBe((999).toLocaleString("uk-UA"));

    tab.destroy();
    container.remove();
  });

  it("lets no page read the dismissal disowned paint its failure over the one that replaced it", async () => {
    const settle: Array<{
      resolve: (stats: HistoryStats) => void;
      reject: (e: Error) => void;
    }> = [];
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve, reject) => settle.push({ resolve, reject })),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(settle).toHaveLength(1);

    tab.releaseResources!();
    tab.resumeResources!();
    expect(settle).toHaveLength(2);

    settle[1].resolve(buildStats({ total_words: 999 }));
    await vi.advanceTimersByTimeAsync(0);
    settle[0].reject(new Error("the backend went away"));
    await vi.advanceTimersByTimeAsync(0);

    expect(
      container.textContent,
      "a read the dismissal disowned may not paint its failure either: the window is back, " +
        "its own read has landed, and an error from before it is not what is on screen",
    ).not.toContain("Failed to load");
    expect(document.getElementById("words-stat-lifetime")?.textContent).toBe(
      (999).toLocaleString("uk-UA"),
    );

    tab.destroy();
    container.remove();
  });

  it("starts no second page read while the first can still paint", async () => {
    const settle: Array<(stats: HistoryStats) => void> = [];
    apiMock.historyStats.mockImplementation(
      () => new Promise<HistoryStats>((resolve) => settle.push(resolve)),
    );

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(settle).toHaveLength(1);

    tab.resumeResources!();

    expect(
      settle,
      "a resume landing while the mount's own read can still paint has nothing to " +
        "repair, and a second whole-page read there is two requests racing one screen",
    ).toHaveLength(1);

    tab.destroy();
    container.remove();
  });

  it("reads nothing at all when it mounts into a dismissed window", async () => {
    apiMock.historyStats.mockResolvedValue(buildStats());

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container, true);
    await vi.advanceTimersByTimeAsync(0);

    expect(
      apiMock.historyStats,
      "the mount's own requests are away before any release can run, so a tab mounted " +
        "into a window nobody can see still costs them",
    ).not.toHaveBeenCalled();
    expect(apiMock.wordsTop).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(30_000);
    expect(apiMock.historyStats).not.toHaveBeenCalled();

    tab.resumeResources!();
    expect(
      apiMock.historyStats,
      "and the show is what pays for them, once",
    ).toHaveBeenCalledTimes(1);

    tab.destroy();
    container.remove();
  });
});

describe("the Words tab while the Settings window stays open", () => {
  it("puts the page back on the next tick when a chained read failed as entries appeared", async () => {
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 0 }));
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 7 }));
    apiMock.historyStats.mockRejectedValueOnce(new Error("the backend went away"));

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("No transcriptions yet");

    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("Failed to load");

    apiMock.historyStats.mockResolvedValue(buildStats({ total_entries: 7, total_words: 4321 }));
    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(0);

    expect(
      document.getElementById("words-stat-lifetime")?.textContent,
      "the window is open and the backend has answered again, so the screen has to come " +
        "back on its own rather than wait for the user to close and reopen it",
    ).toBe((4321).toLocaleString("uk-UA"));

    tab.destroy();
    container.remove();
  });

  it("puts the page back on the next tick when a chained read failed as entries went away", async () => {
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 5 }));
    apiMock.historyStats.mockResolvedValueOnce(buildStats({ total_entries: 0 }));
    apiMock.historyStats.mockRejectedValueOnce(new Error("the backend went away"));

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(document.getElementById("words-stat-entries")).not.toBeNull();

    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("Failed to load");

    apiMock.historyStats.mockResolvedValue(buildStats({ total_entries: 9, total_words: 4321 }));
    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(0);

    expect(
      document.getElementById("words-stat-lifetime")?.textContent,
      "a read that threw with the entry count back at zero paints the same error line as " +
        "any other, and the tick that follows has to replace it like any other",
    ).toBe((4321).toLocaleString("uk-UA"));

    tab.destroy();
    container.remove();
  });

  it("replaces the error line with the empty screen when the recovered history is empty", async () => {
    apiMock.historyStats.mockRejectedValueOnce(new Error("the backend went away"));
    apiMock.historyStats.mockResolvedValue(buildStats({ total_entries: 0 }));

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("Failed to load");

    const spentByTheFailedMount = apiMock.historyStats.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    await vi.advanceTimersByTimeAsync(0);

    expect(
      container.textContent,
      "a read that failed and a history with nothing in it are two different screens, and " +
        "a page that records them as one leaves the error up on a backend that is answering",
    ).toContain("No transcriptions yet");
    expect(
      apiMock.historyStats.mock.calls.length,
      "the whole-page read is the tick's answer in full, so a stats read behind it is a " +
        "second round trip for a page already painted",
    ).toBe(spentByTheFailedMount + 1);

    tab.destroy();
    container.remove();
  });

  it("asks for top words once while the history stays empty, not once every tick", async () => {
    apiMock.historyStats.mockResolvedValue(buildStats({ total_entries: 0 }));

    const container = document.createElement("div");
    document.body.appendChild(container);
    const tab = renderWords(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(container.textContent).toContain("No transcriptions yet");

    await vi.advanceTimersByTimeAsync(60_000);

    expect(
      apiMock.wordsTop,
      "an empty screen carries no figures to patch and no top-words list to fill, so a " +
        "request for one costs a round trip every five seconds and changes nothing",
    ).toHaveBeenCalledTimes(1);

    tab.destroy();
    container.remove();
  });
});
