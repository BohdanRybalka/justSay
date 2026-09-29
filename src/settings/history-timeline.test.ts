// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import type { HistoryDay, HistoryEntry } from "../api";
import type { BuiltRow } from "./history-list";
import { buildEntry } from "./history-page-stub.test-helper";
import {
  createTimelineRows,
  dayKeyOf,
  formatClock,
  formatDayLabel,
  formatDuration,
  formatNumber,
  localDayKey,
} from "./history-timeline";

const NOW = new Date(2026, 7, 1, 12, 0);

function at(day: number, hour: number, minute = 0): string {
  return new Date(2026, 7, day, hour, minute).toISOString();
}

function row(id: string, timestamp: string | null, words = 4): BuiltRow {
  const entry: HistoryEntry = { ...buildEntry(id), timestamp, word_count: words };
  const element = document.createElement("article");
  element.className = "entry";
  element.dataset.id = id;
  return { entry, element };
}

function layout(container: HTMLElement): { head: string; ids: string[] }[] {
  return Array.from(container.querySelectorAll<HTMLElement>(".day-group")).map((group) => ({
    head: group.querySelector(".day-head")!.textContent!,
    ids: Array.from(group.querySelectorAll<HTMLElement>(".entry")).map((el) => el.dataset.id!),
  }));
}

describe("formats fixed to the design whatever the OS locale", () => {
  it("writes thousands with a comma", () => {
    expect(formatNumber(7142)).toBe("7,142");
    expect(formatNumber(117)).toBe("117");
  });

  it("names today, yesterday, then weekday, day and month", () => {
    expect(formatDayLabel("2026-08-01", NOW)).toBe("Today");
    expect(formatDayLabel("2026-07-31", NOW)).toBe("Yesterday");
    expect(formatDayLabel("2026-07-28", NOW)).toBe("Tuesday, 28 July");
  });

  it("adds the year only for a day from another year, and names an unknown day", () => {
    expect(formatDayLabel("2025-12-31", NOW)).toBe("Wednesday, 31 December 2025");
    expect(formatDayLabel(null, NOW)).toBe("Date unknown");
  });

  it("reads Yesterday across a month and a year boundary", () => {
    expect(formatDayLabel("2025-12-31", new Date(2026, 0, 1, 8))).toBe("Yesterday");
  });

  it("writes the clock on 24 hours with leading zeros", () => {
    expect(formatClock(new Date(2026, 7, 1, 18, 26))).toBe("18:26");
    expect(formatClock(new Date(2026, 7, 1, 7, 5))).toBe("07:05");
  });

  it("writes lengths as m:ss, and h:mm:ss from an hour on", () => {
    expect(formatDuration(60)).toBe("1:00");
    expect(formatDuration(14.4)).toBe("0:14");
    expect(formatDuration(48 * 60 + 12)).toBe("48:12");
    expect(formatDuration(3600 + 2 * 60 + 3)).toBe("1:02:03");
  });
});

describe("the local day an entry belongs to", () => {
  const originalTz = process.env.TZ;

  afterEach(() => {
    process.env.TZ = originalTz;
  });

  it("splits at local midnight, not at UTC midnight", () => {
    expect(dayKeyOf({ ...buildEntry("a"), timestamp: at(0, 23, 59) })).toBe("2026-07-31");
    expect(dayKeyOf({ ...buildEntry("b"), timestamp: at(1, 0, 1) })).toBe("2026-08-01");
  });

  it("follows the machine's time zone for the same instant", () => {
    const entry = { ...buildEntry("a"), timestamp: "2026-07-31T20:30:00Z" };

    process.env.TZ = "Asia/Tokyo";
    expect(dayKeyOf(entry)).toBe("2026-08-01");

    process.env.TZ = "America/New_York";
    expect(dayKeyOf(entry)).toBe("2026-07-31");
  });

  it("has no day for a record with no recording time", () => {
    expect(dayKeyOf({ ...buildEntry("a"), timestamp: null })).toBeNull();
    expect(localDayKey(new Date(2026, 0, 5))).toBe("2026-01-05");
  });
});

describe("createTimelineRows — day groups", () => {
  it("puts each page's rows under their day, newest day first, a later page below", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);

    rows.replace([row("a", at(1, 10)), row("b", at(1, 9)), row("c", at(0, 22))], []);
    rows.append([row("d", at(0, 8)), row("e", at(-3, 8))], []);

    expect(layout(container).map((group) => group.ids)).toEqual([["a", "b"], ["c", "d"], ["e"]]);
  });

  it("reads a header from the day's totals and falls back to the cards only for a day it was not told", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);
    const today: HistoryDay = { date: "2026-08-01", recordings: 4, words: 7142 };

    rows.replace([row("a", at(1, 10), 100), row("b", at(0, 10), 1), row("c", at(0, 9), 2)], [today]);

    expect(layout(container).map((group) => group.head)).toEqual([
      "Today·4 recordings·7,142 words",
      "Yesterday·2 recordings·3 words",
    ]);
  });

  it("writes one recording and one word in the singular", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);

    rows.replace([row("a", at(1, 10), 1)], []);

    expect(layout(container)[0].head).toBe("Today·1 recording·1 word");
  });

  it("puts newer rows on top of their day in order, and opens a new day above the rest", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);
    rows.replace([row("old", at(0, 10))], []);

    rows.prepend([row("newest", at(1, 11)), row("newer", at(1, 9))], [
      { date: "2026-08-01", recordings: 2, words: 8 },
    ]);
    rows.prepend([row("latest", at(1, 11, 30))], [{ date: "2026-08-01", recordings: 3, words: 12 }]);

    expect(layout(container)).toEqual([
      { head: "Today·3 recordings·12 words", ids: ["latest", "newest", "newer"] },
      { head: "Yesterday·1 recording·4 words", ids: ["old"] },
    ]);
  });

  it("drops the groups when another lane paints flat rows, and starts over on the next page", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);
    rows.replace([row("a", at(1, 10))], [{ date: "2026-08-01", recordings: 9, words: 9 }]);

    const match = document.createElement("article");
    rows.replaceWith([match]);
    expect(Array.from(container.children)).toEqual([match]);

    rows.replace([row("b", at(1, 10))], []);
    expect(layout(container)).toEqual([{ head: "Today·1 recording·4 words", ids: ["b"] }]);
  });

  it("removes a card, and its day once the day is empty", () => {
    const container = document.createElement("div");
    const rows = createTimelineRows(container, () => NOW);
    const a = row("a", at(1, 10));
    const b = row("b", at(0, 10));
    rows.replace([a, b], []);

    rows.rowRemoved(b.element);
    rows.prepend([row("c", at(1, 11))], []);

    expect(layout(container).map((group) => group.ids)).toEqual([["c", "a"]]);
  });
});
