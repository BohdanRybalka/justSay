import { describe, expect, it } from "vitest";
import { formatCoarseDuration, formatElapsedClock, formatStopwatch } from "./format";

describe("formatElapsedClock — the meeting recording readout", () => {
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

describe("formatStopwatch — the dictation counter", () => {
  it("shows tenths under a minute and minutes above one", () => {
    expect(formatStopwatch(0)).toBe("0.0s");
    expect(formatStopwatch(9.5)).toBe("9.5s");
    expect(formatStopwatch(61.25)).toBe("1:01.2");
  });

  it("truncates the tenth rather than rounding it", () => {
    expect(formatStopwatch(9.79)).toBe("9.7s");
  });

  it("pads the seconds so the readout does not jump width", () => {
    expect(formatStopwatch(65)).toBe("1:05.0");
  });
});

describe("formatCoarseDuration — a total read at a glance", () => {
  it("drops to the largest useful unit", () => {
    expect(formatCoarseDuration(45)).toBe("45 s");
    expect(formatCoarseDuration(3665)).toBe("1 h 1 m");
    expect(formatCoarseDuration(125)).toBe("2 m 5 s");
  });

  it("treats absent, zero and negative totals as zero", () => {
    expect(formatCoarseDuration(0)).toBe("0 m");
    expect(formatCoarseDuration(-10)).toBe("0 m");
  });
});

describe("the three formatters stay distinguishable", () => {
  it("renders the same input three different ways, which is why they are three functions", () => {
    const seconds = 61;
    const rendered = new Set([
      formatStopwatch(seconds),
      formatElapsedClock(seconds),
      formatCoarseDuration(seconds),
    ]);
    expect(rendered.size).toBe(3);
  });
});
