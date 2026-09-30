// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FillerNote, Insights, TopWordsResponse, WordCount } from "../../api";

const { apiMock, imageMock } = vi.hoisted(() => ({
  apiMock: { insights: vi.fn(), wordsTop: vi.fn() },
  imageMock: { savedCardPng: vi.fn(), copyImage: vi.fn(), saveImage: vi.fn() },
}));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

vi.mock("./insights-image", () => imageMock);

import { greetingFor, imageName, paceLine, renderInsights, savedCardView, savedPhrase } from "./insights";

function spokenDays(words: number[]): Insights["days"] {
  return words.map((value, index) => ({ date: `2026-09-${String(index + 30 - words.length).padStart(2, "0")}`, words: value }));
}

function buildInsights(overrides: {
  today?: Partial<Insights["today"]>;
  month?: Partial<Insights["month"]>;
  streak?: Partial<Insights["streak"]>;
  days?: Insights["days"];
  previous_period_words?: number;
  meetings_week?: Insights["meetings_week"];
} = {}): Insights {
  return {
    days: overrides.days ?? spokenDays([...Array(28).fill(100), 300]),
    previous_period_words: overrides.previous_period_words ?? 2000,
    today: { words: 1212, recordings: 8, ...overrides.today },
    month: {
      words: 10666,
      recordings: 116,
      speaking_seconds: 5400,
      typing_seconds: 16020,
      pace_wpm: 118,
      ...overrides.month,
    },
    streak: { current_days: 8, longest_days: 8, ...overrides.streak },
    vocabulary: 2140,
    peak_hour: { hour: 18, share: 0.31 },
    longest: { seconds: 393, words: 715 },
    meetings_week: overrides.meetings_week ?? { count: 3, seconds: 7860 },
  };
}

const FILLER_WORDS = new Set(["so", "like", "just"]);
const FAVOURITES: WordCount[] = ["or", "by", "so", "mean", "example", "like", "which", "will", "just", "overall"].map(
  (word, index) => ({ word, count: 50 - index * 4, is_filler: FILLER_WORDS.has(word) }),
);
const NOTE: FillerNote = { word: "so", count: 42, minutes_between: 2.5, fillers_in_top: 3, top_size: 10 };
const FILLERS_ONLY: TopWordsResponse = {
  items: FAVOURITES.filter((item) => item.is_filler),
  note: NOTE,
};

const viewer = { name: "Bohdan Rybalka", shortcut: "Ctrl+Alt+KeyV" };
let container: HTMLElement;

async function mount(figures: Insights, windowHidden = false, favourites: WordCount[] = FAVOURITES) {
  apiMock.insights.mockResolvedValue(figures);
  apiMock.wordsTop.mockResolvedValue({ items: favourites, note: favourites.some((item) => item.is_filler) ? NOTE : null });
  const tab = renderInsights(container, viewer, windowHidden);
  await vi.advanceTimersByTimeAsync(0);
  return tab;
}

const text = (selector: string) => container.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 8, 29, 20, 15));
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => {});
  container = document.createElement("div");
  document.body.append(container);
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("greetingFor", () => {
  it("says morning from 5, afternoon from 12 and evening from 17 until 5", () => {
    expect([4, 5, 11, 12, 16, 17, 23, 0].map(greetingFor)).toEqual([
      "Good evening", "Good morning", "Good morning", "Good afternoon",
      "Good afternoon", "Good evening", "Good evening", "Good evening",
    ]);
  });
});

describe("savedPhrase", () => {
  it("names the time saved by the step it has reached", () => {
    expect([0, 9, 10, 29, 30, 89, 90, 239, 240, 479, 480].map(savedPhrase)).toEqual([
      "A few minutes back.", "A few minutes back.", "About a coffee break.", "About a coffee break.",
      "About a long lunch.", "About a long lunch.", "Roughly one free afternoon.",
      "Roughly one free afternoon.", "About a whole day off.", "About a whole day off.",
      "More than a working day.",
    ]);
  });
});

describe("paceLine", () => {
  it("compares the pace with typing in whole times", () => {
    expect([0.5, 1.4, 1.6, 2.95, 4.4].map(paceLine)).toEqual([
      "about your typing speed", "about your typing speed", "twice your typing",
      "three times your typing", "4 times your typing",
    ]);
  });
});

describe("the Insights panel", () => {
  it("greets by first name and counts today's dictation with the shortcut", async () => {
    await mount(buildInsights());

    expect(text(".panel-title")).toBe("Good evening, Bohdan");
    expect(text("#insights-today")).toBe(
      "1,212 words today across 8 recordings. Hold Ctrl + Alt + V anywhere to add more.",
    );
  });

  it("shows the time saved as typing minus speaking, with both bars", async () => {
    await mount(buildInsights());

    expect(text(".saved-card-value")).toBe("2 h 57 m");
    expect(text(".saved-card-note")).toBe(
      "That's what typing these words by hand would have cost you. Roughly one free afternoon.",
    );
    expect(text(".saved-compare")).toBe("Typing4:27 Speaking1:30");
    const bars = [...container.querySelectorAll<HTMLElement>(".saved-compare-track i")].map((bar) => bar.style.width);
    expect(bars).toEqual(["100%", "33.7%"]);
  });

  it("shows the month's words, the streak and the pace below the divider", async () => {
    await mount(buildInsights());

    const figures = [...container.querySelectorAll(".saved-figures > div")].map((figure) =>
      figure.textContent!.replace(/\s+/g, " ").trim(),
    );
    expect(figures).toEqual([
      "10,666words in 116 recordings",
      "8 daysstreak · your longest",
      "118 wpmthree times your typing",
    ]);
  });

  it("calls a streak the longest only when it is and runs past one day", async () => {
    await mount(buildInsights({ streak: { current_days: 3, longest_days: 5 } }));
    expect(text(".saved-figures > div:nth-child(2)")).toBe("3 daysstreak");

    container.innerHTML = "";
    await mount(buildInsights({ streak: { current_days: 1, longest_days: 1 } }));
    expect(text(".saved-figures > div:nth-child(2)")).toBe("1 daystreak");
  });

  it("never shows a negative saving when speaking took longer than typing", async () => {
    await mount(buildInsights({ month: { typing_seconds: 900, speaking_seconds: 1200, pace_wpm: 30 } }));

    expect(text(".saved-card-value")).toBe("0 m");
    expect(text(".saved-card-note")).toContain("A few minutes back.");
  });

  it("holds the bars back under ten minutes of speaking and the pace without any", async () => {
    await mount(buildInsights({ month: { speaking_seconds: 599 } }));
    expect(container.querySelector(".saved-compare")).toBeNull();

    container.innerHTML = "";
    await mount(buildInsights({ month: { speaking_seconds: 0, typing_seconds: 0, pace_wpm: null } }));
    expect(container.querySelectorAll(".saved-figures > div")).toHaveLength(2);
  });

  it("asks for a first dictation today, and leaves the card out for an empty month", async () => {
    await mount(buildInsights({ today: { words: 0, recordings: 0 }, month: { recordings: 0 } }));

    expect(text("#insights-today")).toBe("Hold Ctrl + Alt + V anywhere and talk.");
    expect(container.querySelector(".saved-card")).toBeNull();
  });

  it("draws the last 30 days under the card, and nothing before a first dictation", async () => {
    await mount(buildInsights());
    expect(apiMock.insights).toHaveBeenCalledWith(30);
    expect(container.querySelectorAll(".chart-bar")).toHaveLength(29);
    expect(text(".chart-delta")).toBe("+55% vs previous 30 days");

    container.innerHTML = "";
    await mount(buildInsights({ month: { recordings: 0 }, days: spokenDays(Array(30).fill(0)), previous_period_words: 0 }));
    expect(container.querySelector(".chart")).toBeNull();
  });

  it("keeps the chart for last month's words when this month has none yet", async () => {
    await mount(buildInsights({ month: { recordings: 0 }, days: spokenDays([...Array(29).fill(0), 40]) }));

    expect(container.querySelector(".saved-card")).toBeNull();
    expect(container.querySelectorAll(".chart-bar")).toHaveLength(30);
  });

  it("switches to 7 days by redrawing the chart alone", async () => {
    await mount(buildInsights());
    const card = container.querySelector(".saved-card");
    const todayLine = container.querySelector("#insights-today")!.innerHTML;
    apiMock.insights.mockResolvedValue(
      buildInsights({ today: { words: 9 }, days: spokenDays([1, 2, 3, 4, 5, 6, 7]), previous_period_words: 0 }),
    );

    [...container.querySelectorAll<HTMLButtonElement>(".chart-range button")].find((b) => b.textContent === "7d")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.insights).toHaveBeenLastCalledWith(7);
    expect(container.querySelectorAll(".chart-bar")).toHaveLength(7);
    expect(container.querySelector<HTMLElement>(".chart-delta")!.hidden).toBe(true);
    expect(container.querySelector(".saved-card")).toBe(card);
    expect(container.querySelector("#insights-today")!.innerHTML).toBe(todayLine);
    expect(container.querySelector(".chart-range [aria-pressed=true]")!.textContent).toBe("7d");
  });

  it("keeps the chart and its switch when a quiet week comes back empty", async () => {
    const tab = await mount(buildInsights());
    const empty = buildInsights({ month: { recordings: 0 }, days: spokenDays(Array(7).fill(0)), previous_period_words: 0 });
    apiMock.insights.mockResolvedValue(empty);
    container.querySelector<HTMLButtonElement>(".chart-range button")!.click();
    await vi.advanceTimersByTimeAsync(0);

    tab.resumeResources!();
    await vi.advanceTimersByTimeAsync(0);

    expect(container.querySelectorAll(".chart-bar")).toHaveLength(7);
    expect(container.querySelector(".chart-range [aria-pressed=true]")!.textContent).toBe("7d");
  });

  it("lets a switch and a returning window both land, then redraws the chosen span", async () => {
    const tab = await mount(buildInsights());
    const answers: ((figures: Insights) => void)[] = [];
    apiMock.insights.mockImplementation(() => new Promise<Insights>((resolve) => answers.push(resolve)));

    tab.resumeResources!();
    container.querySelector<HTMLButtonElement>(".chart-range button")!.click();
    answers[1](buildInsights({ days: spokenDays([1, 2, 3, 4, 5, 6, 7]) }));
    await vi.advanceTimersByTimeAsync(0);
    answers[0](buildInsights({ today: { words: 5 } }));
    await vi.advanceTimersByTimeAsync(0);

    expect(text("#insights-today")).toContain("5 words");
    expect(apiMock.insights.mock.calls.map(([days]) => days)).toEqual([30, 30, 7, 7]);
    answers[2](buildInsights({ days: spokenDays([9, 9, 9, 9, 9, 9, 9]) }));
    await vi.advanceTimersByTimeAsync(0);
    expect(container.querySelectorAll(".chart-bar")).toHaveLength(7);
    expect(container.querySelector(".chart-range [aria-pressed=true]")!.textContent).toBe("7d");
  });

  it("keeps the chosen span when the window comes back", async () => {
    const tab = await mount(buildInsights());
    container.querySelector<HTMLButtonElement>(".chart-range button")!.click();
    await vi.advanceTimersByTimeAsync(0);

    tab.resumeResources!();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.insights).toHaveBeenLastCalledWith(7);
    expect(container.querySelector(".chart-range [aria-pressed=true]")!.textContent).toBe("7d");
  });

  it("greets without a name when none is known", async () => {
    apiMock.insights.mockResolvedValue(buildInsights());
    apiMock.wordsTop.mockResolvedValue({ items: FAVOURITES, note: NOTE });
    renderInsights(container, { ...viewer, name: " " }, false);

    expect(text(".panel-title")).toBe("Good evening");
  });

  it("reads again when the window comes back, and drops a read the window left behind", async () => {
    const tab = await mount(buildInsights());
    let answer: (figures: Insights) => void = () => {};
    apiMock.insights.mockImplementation(() => new Promise<Insights>((resolve) => (answer = resolve)));

    tab.resumeResources!();
    tab.releaseResources!();
    answer(buildInsights({ today: { words: 5 } }));
    await vi.advanceTimersByTimeAsync(0);
    expect(text("#insights-today")).toContain("1,212 words");

    tab.resumeResources!();
    answer(buildInsights({ today: { words: 5 } }));
    await vi.advanceTimersByTimeAsync(0);
    expect(text("#insights-today")).toContain("5 words");
    expect(apiMock.insights).toHaveBeenCalledTimes(3);
  });

  it("reads nothing while mounted in a hidden window", async () => {
    await mount(buildInsights(), true);

    expect(apiMock.insights).not.toHaveBeenCalled();
  });

  it("tells four facts about your voice under the chart", async () => {
    await mount(buildInsights());

    const facts = [...container.querySelectorAll(".fact-card")].map((card) => card.textContent!.replace(/\s+/g, " ").trim());
    expect(facts).toEqual([
      "2,140 different wordsAcross everything you've said so far.",
      "You talk most at 6 PMAlmost a third of everything you dictate happens between 5 and 7.",
      "6:33Your longest run without stopping. It became 715 words.",
      "3 meetingscaptured this week — 2 h 11 m of talk turned into notes you can search.",
    ]);
    expect(container.querySelector(".chart")!.compareDocumentPosition(container.querySelector(".fact-grid")!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("puts the three favourite words on a podium and ranks the next seven as bars", async () => {
    apiMock.wordsTop.mockClear();
    await mount(buildInsights());

    expect(apiMock.wordsTop).toHaveBeenCalledWith(10, "all");
    const podium = [...container.querySelectorAll(".podium-item")].map((item) => item.textContent!.replace(/\s+/g, " ").trim());
    expect(podium).toEqual(["1 or 50 times", "2 by 46 times", "3 so 42 times · filler"]);
    const rows = [...container.querySelectorAll(".word-row")].map((row) => row.textContent!.replace(/\s+/g, " ").trim());
    expect(rows[0]).toBe("4 mean 38");
    expect(rows).toHaveLength(7);
    expect(container.querySelector(".fact-grid")!.compareDocumentPosition(container.querySelector(".favourite-words")!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("grows each bar from zero to its count against the most said word", async () => {
    await mount(buildInsights());
    const fills = () => [...container.querySelectorAll<HTMLElement>(".word-row-track i")].map((fill) => fill.style.width);

    expect(new Set(fills())).toEqual(new Set(["0px"]));
    await vi.advanceTimersByTimeAsync(20);

    expect(fills()[0]).toBe("76%");
    expect(fills()[6]).toBe("28%");
  });

  it("leaves out the meetings card without meetings, and the words before any", async () => {
    await mount(buildInsights({ meetings_week: { count: 0, seconds: 0 } }), false, []);

    expect(container.querySelectorAll(".fact-card")).toHaveLength(3);
    expect(container.querySelector(".favourite-words")).toBeNull();
  });

  it("shows the voice and the words of someone who spoke before this chart began", async () => {
    await mount(buildInsights({ month: { recordings: 0 }, days: spokenDays(Array(30).fill(0)), previous_period_words: 0 }));

    expect(container.querySelector(".chart")).toBeNull();
    expect(container.querySelectorAll(".fact-card")).toHaveLength(4);
    expect(container.querySelectorAll(".podium-item")).toHaveLength(3);
  });

  it("offers Try again when the words cannot be read", async () => {
    apiMock.wordsTop.mockRejectedValueOnce(new Error("HTTP 503"));
    apiMock.insights.mockResolvedValue(buildInsights());
    renderInsights(container, viewer, false);
    await vi.advanceTimersByTimeAsync(0);

    expect(text(".panel-error")).toBe("Insights could not be read. Try again");
  });

  it("offers Try again on a failed read, and the retry fills the panel", async () => {
    apiMock.wordsTop.mockResolvedValue({ items: FAVOURITES, note: NOTE });
    apiMock.insights.mockRejectedValueOnce(new Error("HTTP 503"));
    renderInsights(container, viewer, false);
    await vi.advanceTimersByTimeAsync(0);

    expect(text(".panel-error")).toBe("Insights could not be read. Try again");
    apiMock.insights.mockResolvedValue(buildInsights());
    container.querySelector<HTMLButtonElement>(".panel-error button")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(text(".saved-card-value")).toBe("2 h 57 m");
  });

  it("flags fillers in clay and says how often you say the top one", async () => {
    await mount(buildInsights());

    expect([...container.querySelectorAll(".podium-item--filler .podium-word")].map((word) => word.textContent)).toEqual(["so"]);
    const flagged = [...container.querySelectorAll(".word-row")].filter((row) => row.querySelector(".chip"));
    expect(flagged.map((row) => row.querySelector(".word-row-word")!.textContent!.replace(/\s+/g, " ").trim())).toEqual([
      "like filler",
      "just filler",
    ]);
    expect(container.querySelectorAll(".word-row-fill--filler")).toHaveLength(2);
    expect(text(".words-note")).toBe(
      "You said so 42 times — about once every two and a half minutes. Three of your top ten are filler words.",
    );
  });

  it("filters the podium and the list to fillers and back, leaving the rest of the panel", async () => {
    await mount(buildInsights());
    const chart = container.querySelector(".chart");
    apiMock.wordsTop.mockResolvedValue(FILLERS_ONLY);

    container.querySelectorAll<HTMLButtonElement>(".words-filter button")[1].click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.wordsTop).toHaveBeenLastCalledWith(10, "fillers");
    expect([...container.querySelectorAll(".podium-word")].map((word) => word.textContent)).toEqual(["so", "like", "just"]);
    expect(container.querySelector(".word-list")).toBeNull();
    expect(container.querySelector(".chart")).toBe(chart);
    expect(text(".words-note")).toContain("You said so 42 times");

    apiMock.wordsTop.mockResolvedValue({ items: FAVOURITES, note: NOTE });
    container.querySelectorAll<HTMLButtonElement>(".words-filter button")[0].click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.wordsTop).toHaveBeenLastCalledWith(10, "all");
    expect(container.querySelectorAll(".podium-item")).toHaveLength(3);
    expect(container.querySelectorAll(".word-row")).toHaveLength(7);
  });

  it("keeps the fillers filter across reads, and falls back to all once no filler is left", async () => {
    const tab = await mount(buildInsights());
    apiMock.wordsTop.mockResolvedValue(FILLERS_ONLY);
    container.querySelectorAll<HTMLButtonElement>(".words-filter button")[1].click();
    await vi.advanceTimersByTimeAsync(0);

    tab.resumeResources!();
    await vi.advanceTimersByTimeAsync(0);
    expect(apiMock.wordsTop).toHaveBeenLastCalledWith(10, "fillers");
    expect(container.querySelector(".words-filter [aria-pressed=true]")!.textContent).toBe("Fillers");

    const plain = FAVOURITES.filter((item) => !item.is_filler);
    apiMock.wordsTop.mockImplementation(async (_limit: number, filter: string) =>
      filter === "fillers" ? { items: [], note: null } : { items: plain, note: null },
    );
    tab.resumeResources!();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.wordsTop).toHaveBeenLastCalledWith(10, "all");
    expect(container.querySelectorAll(".podium-item")).toHaveLength(3);
    expect(container.querySelector(".words-filter button")).toBeNull();
    expect(container.querySelector(".words-note")).toBeNull();
  });

  it("drops a filter answer that a later switch has overtaken", async () => {
    await mount(buildInsights());
    let answerFillers: (top: TopWordsResponse) => void = () => {};
    apiMock.wordsTop.mockImplementationOnce(() => new Promise<TopWordsResponse>((resolve) => (answerFillers = resolve)));
    const [all, fillers] = container.querySelectorAll<HTMLButtonElement>(".words-filter button");

    fillers.click();
    apiMock.wordsTop.mockResolvedValue({ items: FAVOURITES, note: NOTE });
    all.click();
    await vi.advanceTimersByTimeAsync(0);
    answerFillers(FILLERS_ONLY);
    await vi.advanceTimersByTimeAsync(0);

    expect(container.querySelectorAll(".word-row")).toHaveLength(7);
  });

  it("reads the words again when the filter changed while the panel was reading", async () => {
    const tab = await mount(buildInsights());
    let answerAll: (top: TopWordsResponse) => void = () => {};
    apiMock.wordsTop.mockImplementationOnce(() => new Promise<TopWordsResponse>((resolve) => (answerAll = resolve)));
    tab.resumeResources!();
    apiMock.wordsTop.mockResolvedValue(FILLERS_ONLY);

    container.querySelectorAll<HTMLButtonElement>(".words-filter button")[1].click();
    await vi.advanceTimersByTimeAsync(0);
    answerAll({ items: FAVOURITES, note: NOTE });
    await vi.advanceTimersByTimeAsync(0);

    expect(container.querySelector(".words-filter [aria-pressed=true]")!.textContent).toBe("Fillers");
    expect([...container.querySelectorAll(".podium-word")].map((word) => word.textContent)).toEqual(["so", "like", "just"]);
  });

  it("copies the card as an image and says so", async () => {
    await mount(buildInsights());
    const png = new Uint8Array([137, 80]);
    imageMock.savedCardPng.mockResolvedValue(png);
    imageMock.copyImage.mockResolvedValue(undefined);

    container.querySelector<HTMLButtonElement>(".share-month .btn-primary")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(imageMock.savedCardPng).toHaveBeenCalledWith(savedCardView(buildInsights()));
    expect(imageMock.copyImage).toHaveBeenCalledWith(png);
    expect(text(".share-month-status")).toBe("Image copied — paste it anywhere");
  });

  it("saves the card under this month's name, and says when copying failed", async () => {
    await mount(buildInsights());
    const png = new Uint8Array([137, 80]);
    imageMock.savedCardPng.mockResolvedValue(png);
    imageMock.saveImage.mockResolvedValue(false);
    const [share, download] = container.querySelectorAll<HTMLButtonElement>(".share-month button");

    download.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(imageMock.saveImage).toHaveBeenCalledWith(png, "JustSay-September-2026.png");
    expect(text(".share-month-status")).toBe("");

    imageMock.copyImage.mockRejectedValue(new Error("the clipboard stayed busy"));
    share.click();
    await vi.advanceTimersByTimeAsync(0);
    expect(text(".share-month-status")).toBe("Couldn't copy the image. Try again");
    expect(share.disabled).toBe(false);
  });

  it("offers no image before a first recording this month", async () => {
    await mount(buildInsights({ month: { recordings: 0 } }));

    expect(container.querySelector(".share-month")).toBeNull();
  });
});

describe("the saved card as data", () => {
  it("holds what the card shows, bars measured against the longer time", () => {
    const view = savedCardView(buildInsights());

    expect(view.value).toBe("2 h 57 m");
    expect(view.compare.map(({ label, fraction, dim }) => [label, fraction, dim])).toEqual([
      ["Typing", 1, true],
      ["Speaking", 90 / 267, false],
    ]);
    expect(view.figures.map((figure) => figure.label)).toEqual(["words in 116 recordings", "streak · your longest", "three times your typing"]);
  });

  it("names the image after the month", () => {
    expect(imageName(new Date(2026, 0, 3))).toBe("JustSay-January-2026.png");
  });
});

