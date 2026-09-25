import { describe, expect, it } from "vitest";
import { levelFromDb } from "./level";

describe("levelFromDb", () => {
  it.each([
    [-90, 0],
    [-60, 0],
    [-30, 0.5],
    [0, 1],
    [6, 1],
  ])("reads %d dB as %d of the meter", (db, level) => {
    expect(levelFromDb(db)).toBeCloseTo(level);
  });
});
