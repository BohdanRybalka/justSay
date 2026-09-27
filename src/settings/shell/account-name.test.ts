import { describe, expect, it } from "vitest";
import { displayName, initialsOf } from "./account-name";

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

describe("displayName", () => {
  it.each([
    ["Ada", "Bohdan Rybalka", "Ada"],
    ["  Ada Lovelace ", "Bohdan Rybalka", "Ada Lovelace"],
    ["", "Bohdan Rybalka", "Bohdan Rybalka"],
    ["   ", "Bohdan Rybalka", "Bohdan Rybalka"],
    ["", "", ""],
  ])("chosen %j over the computer's %j reads %j", (chosen, osName, shown) => {
    expect(displayName(chosen, osName)).toBe(shown);
  });
});
