import { describe, expect, it } from "vitest";
import type { FillerNote } from "../../api";
import { fillerNote, frequencyPhrase, numberWord } from "./insights-words";

const note = (overrides: Partial<FillerNote> = {}): FillerNote => ({
  word: "like",
  count: 37,
  minutes_between: 2.5,
  fillers_in_top: 6,
  top_size: 10,
  ...overrides,
});

describe("the filler note", () => {
  it("reads as the design writes it", () => {
    expect(fillerNote(note())).toBe(
      "You said <b>like</b> 37 times — about once every two and a half minutes. Six of your top ten are filler words.",
    );
  });

  it("says once, twice, one filler and all of them in plain words", () => {
    expect(fillerNote(note({ count: 1, fillers_in_top: 1 }))).toBe(
      "You said <b>like</b> once — about once every two and a half minutes. One of your top ten is a filler word.",
    );
    expect(fillerNote(note({ count: 2, fillers_in_top: 7, top_size: 7 }))).toContain("<b>like</b> twice");
    expect(fillerNote(note({ fillers_in_top: 7, top_size: 7 }))).toContain("All of your top seven are filler words.");
  });

  it("leaves out what it cannot say", () => {
    expect(fillerNote(note({ minutes_between: null, fillers_in_top: 0 }))).toBe("You said <b>like</b> 37 times.");
    expect(fillerNote(note({ fillers_in_top: 1, top_size: 1 }))).toBe(
      "You said <b>like</b> 37 times — about once every two and a half minutes.",
    );
  });

  it("escapes the word", () => {
    expect(fillerNote(note({ word: "<i>" }))).toContain("<b>&lt;i&gt;</b>");
  });
});

describe("how often a filler comes", () => {
  it.each([
    [0.5, "more than once a minute"],
    [0.8, "about once a minute"],
    [1.2, "about once a minute"],
    [2.5, "about once every two and a half minutes"],
    [3.1, "about once every three minutes"],
    [75, "about once every 75 minutes"],
    [89.6, "about once every 89 and a half minutes"],
    [90, "about once every one and a half hours"],
    [130, "about once every two hours"],
  ])("%s minutes reads %s", (minutes, phrase) => {
    expect(frequencyPhrase(minutes)).toBe(phrase);
  });
});

describe("numbers in words", () => {
  it("writes up to twenty in words and the rest as figures", () => {
    expect([0, 6, 10, 20, 21, 1500].map(numberWord)).toEqual(["zero", "six", "ten", "twenty", "21", "1,500"]);
  });
});
