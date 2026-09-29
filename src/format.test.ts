import { describe, expect, it } from "vitest";
import { formatCoarseDuration, formatElapsedClock, formatHoursClock, wholeMinutes } from "./format";

describe("formatElapsedClock — the dictation and meeting clock", () => {
  it("counts the elapsed time up in minutes and seconds", () => {
    expect(formatElapsedClock(0)).toBe("0:00");
    expect(formatElapsedClock(9.7)).toBe("0:09");
    expect(formatElapsedClock(61)).toBe("1:01");
    expect(formatElapsedClock(3600)).toBe("60:00");
  });

  it("never renders a negative duration from a clock that moved backwards", () => {
    expect(formatElapsedClock(-3)).toBe("0:00");
  });
});

describe("formatCoarseDuration — a total read at a glance", () => {
  it("shows hours and minutes, or minutes alone under an hour", () => {
    expect(formatCoarseDuration(10620)).toBe("2 h 57 m");
    expect(formatCoarseDuration(3600)).toBe("1 h 0 m");
    expect(formatCoarseDuration(125)).toBe("2 m");
  });

  it("treats absent, zero and negative totals as zero", () => {
    expect(formatCoarseDuration(0)).toBe("0 m");
    expect(formatCoarseDuration(-10)).toBe("0 m");
  });
});

describe("formatHoursClock — a long total beside another", () => {
  it("writes hours and two-digit minutes", () => {
    expect(formatHoursClock(16020)).toBe("4:27");
    expect(formatHoursClock(300)).toBe("0:05");
    expect(formatHoursClock(0)).toBe("0:00");
  });
});

describe("wholeMinutes", () => {
  it("rounds to the nearest minute and never goes below zero", () => {
    expect(wholeMinutes(89)).toBe(1);
    expect(wholeMinutes(90)).toBe(2);
    expect(wholeMinutes(-30)).toBe(0);
  });
});

describe("the formatters stay distinguishable", () => {
  it("render the same input different ways, which is why they are separate functions", () => {
    expect(formatElapsedClock(3600)).not.toBe(formatCoarseDuration(3600));
    expect(formatHoursClock(3600)).not.toBe(formatElapsedClock(3600));
  });
});
