// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChartSpan } from "../../api";
import { dayLabel, deltaLabel, isWeekend, mountWordsChart, type ChartFigures } from "./insights-chart";

const DESIGN_WORDS = [180, 0, 240, 410, 120, 0, 0, 320, 505, 290, 150, 0, 470, 610, 380, 220, 0, 0,
  540, 700, 430, 260, 90, 0, 380, 820, 560, 340, 980, 1212];

function daysEnding(last: Date, words: number[]): ChartFigures["days"] {
  return words.map((value, index) => {
    const day = new Date(last.getFullYear(), last.getMonth(), last.getDate() - (words.length - 1 - index));
    const iso = `${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, "0")}-${String(day.getDate()).padStart(2, "0")}`;
    return { date: iso, words: value };
  });
}

let host: HTMLElement;

function draw(words: number[], previous = 0, span: ChartSpan = 30) {
  const chart = mountWordsChart(host, span, () => {});
  chart.draw({ days: daysEnding(new Date(2026, 8, 29), words), previous_period_words: previous }, span);
  return chart;
}

const bars = () => [...host.querySelectorAll<HTMLElement>(".chart-bar")];
const text = (selector: string) => host.querySelector(selector)?.textContent?.replace(/\s+/g, " ").trim();

beforeEach(() => {
  vi.useFakeTimers();
  host = document.createElement("div");
  document.body.append(host);
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("isWeekend", () => {
  it("dims Saturdays and Sundays by the calendar, not by position", () => {
    expect(["2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-03-01"].map(isWeekend)).toEqual([
      false, true, true, false, true,
    ]);
  });
});

describe("dayLabel", () => {
  it("writes the day and the short month", () => {
    expect([dayLabel("2026-07-01"), dayLabel("2026-12-31")]).toEqual(["1 Jul", "31 Dec"]);
  });
});

describe("deltaLabel", () => {
  it("rounds to a whole percent and signs the change", () => {
    expect([
      deltaLabel(134, 100, 30), deltaLabel(66, 100, 30), deltaLabel(1005, 1000, 7),
      deltaLabel(1004, 1000, 7), deltaLabel(996, 1000, 7),
    ]).toEqual([
      "+34% vs previous 30 days", "−34% vs previous 30 days", "+1% vs previous 7 days",
      "0% vs previous 7 days", "0% vs previous 7 days",
    ]);
  });

  it("has nothing to compare with an empty previous period", () => {
    expect(deltaLabel(500, 0, 30)).toBeNull();
  });
});

describe("the words-per-day chart", () => {
  it("grows one bar per day from zero, fourteen milliseconds apart", async () => {
    draw(DESIGN_WORDS);
    const heights = () => bars().map((bar) => bar.querySelector("i")!.style.height);

    expect(bars()).toHaveLength(30);
    expect(new Set(heights())).toEqual(new Set(["0px"]));
    await vi.advanceTimersByTimeAsync(20);
    expect(heights().filter((height) => height !== "0px")).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(29 * 14);
    expect(heights().slice(-2)).toEqual(["72.19%", "89.29%"]);
    expect(heights()[1]).toBe("3%");
  });

  it("dims the real weekend days", () => {
    draw(DESIGN_WORDS);

    const weekends = bars().flatMap((bar, index) => (bar.classList.contains("chart-bar--weekend") ? [index] : []));
    expect(weekends).toEqual([5, 6, 12, 13, 19, 20, 26, 27]);
  });

  it("rings the best day and names it in a callout centred on its bar", () => {
    draw(DESIGN_WORDS);

    expect(bars().findIndex((bar) => bar.classList.contains("chart-bar--best"))).toBe(29);
    expect(text(".chart-callout")).toBe("1,212 words · 29 Sep");
    const callout = host.querySelector<HTMLElement>(".chart-callout")!;
    expect(callout.style.getPropertyValue("--center")).toBe("98.33");
    expect(callout.style.bottom).toBe("89.29%");
  });

  it("gives no best day until two days have words", () => {
    draw([...Array(29).fill(0), 117]);

    expect(host.querySelector(".chart-callout")).toBeNull();
    expect(host.querySelector(".chart-bar--best")).toBeNull();
  });

  it("draws the average over every day of the span, empty ones included", () => {
    draw(DESIGN_WORDS);

    expect(text(".chart-average")).toBe("avg 340");
    expect(host.querySelector<HTMLElement>(".chart-average")!.style.bottom).toBe("25.05%");
  });

  it("leaves the average out when nothing was said", () => {
    draw(Array(7).fill(0), 0, 7);

    expect(host.querySelector(".chart-average")).toBeNull();
    expect(bars()).toHaveLength(7);
  });

  it("shows a bar's words and date on focus and hides them on blur", () => {
    draw(DESIGN_WORDS);
    const tip = host.querySelector<HTMLElement>(".chart-tip")!;

    bars()[19].dispatchEvent(new FocusEvent("focus"));
    expect(tip.textContent).toBe("700 words19 Sep");
    expect(tip.classList.contains("chart-tip--visible")).toBe(true);
    expect(tip.style.getPropertyValue("--center")).toBe("65.00");

    bars()[19].dispatchEvent(new FocusEvent("blur"));
    expect(tip.classList.contains("chart-tip--visible")).toBe(false);
  });

  it("writes three dates under the axis: first, middle, last", () => {
    draw(DESIGN_WORDS);

    expect([...host.querySelectorAll(".chart-axis span")].map((label) => label.textContent)).toEqual([
      "31 Aug", "15 Sep", "29 Sep",
    ]);
  });

  it("compares with the previous period, and hides the chip without one", () => {
    draw(DESIGN_WORDS, 7600);
    expect(text(".chart-delta")).toBe("+34% vs previous 30 days");

    host.innerHTML = "";
    draw(DESIGN_WORDS, 0);
    expect(host.querySelector<HTMLElement>(".chart-delta")!.hidden).toBe(true);
  });
});
