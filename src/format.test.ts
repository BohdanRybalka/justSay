import { describe, expect, it } from "vitest";
import { formatCoarseDuration, formatElapsedClock } from "./format";

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

describe("the two formatters stay distinguishable", () => {
  it("renders the same input two different ways, which is why they are two functions", () => {
    expect(formatElapsedClock(61)).not.toBe(formatCoarseDuration(61));
  });
});
