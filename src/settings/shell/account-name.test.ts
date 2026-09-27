import { describe, expect, it } from "vitest";
import { initialsOf } from "./account-name";

describe("initialsOf", () => {
  it.each([
    ["Bohdan Rybalka", "BR"],
    ["Богдан Рибалка", "БР"],
    ["admin", "A"],
    ["ada lovelace byron", "AL"],
    ["  Bohdan   Rybalka ", "BR"],
    ["", ""],
  ])("turns %j into %j", (name, initials) => {
    expect(initialsOf(name)).toBe(initials);
  });
});
