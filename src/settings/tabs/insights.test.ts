// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Insights } from "../../api";

const { apiMock } = vi.hoisted(() => ({ apiMock: { insights: vi.fn() } }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

import { greetingFor, paceLine, renderInsights, savedPhrase } from "./insights";

function buildInsights(overrides: {
  today?: Partial<Insights["today"]>;
  month?: Partial<Insights["month"]>;
  streak?: Partial<Insights["streak"]>;
} = {}): Insights {
  return {
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
  };
}

const viewer = { name: "Bohdan Rybalka", shortcut: "Ctrl+Alt+KeyV" };
let container: HTMLElement;

async function mount(figures: Insights, windowHidden = false) {
  apiMock.insights.mockResolvedValue(figures);
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

  it("greets without a name when none is known", async () => {
    apiMock.insights.mockResolvedValue(buildInsights());
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

  it("offers Try again on a failed read, and the retry fills the panel", async () => {
    apiMock.insights.mockRejectedValueOnce(new Error("HTTP 503"));
    renderInsights(container, viewer, false);
    await vi.advanceTimersByTimeAsync(0);

    expect(text(".panel-error")).toBe("Insights could not be read. Try again");
    apiMock.insights.mockResolvedValue(buildInsights());
    container.querySelector<HTMLButtonElement>(".panel-error button")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(text(".saved-card-value")).toBe("2 h 57 m");
  });
});
