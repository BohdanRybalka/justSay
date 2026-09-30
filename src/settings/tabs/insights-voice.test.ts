// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { hourLabel, sharePhrase, voiceFacts } from "./insights-voice";

const lines = (html: string) => {
  const host = document.createElement("div");
  host.innerHTML = html;
  return [...host.querySelectorAll(".fact-card")].map((card) => card.textContent!.replace(/\s+/g, " ").trim());
};

const quiet = { vocabulary: 0, peak_hour: null, longest: null, meetings_week: { count: 0, seconds: 0 } };

describe("hourLabel", () => {
  it("writes the hour on a 12-hour clock", () => {
    expect([0, 1, 11, 12, 13, 18, 23].map(hourLabel)).toEqual(["12 AM", "1 AM", "11 AM", "12 PM", "1 PM", "6 PM", "11 PM"]);
  });
});

describe("sharePhrase", () => {
  it("names a share by the phrase that is true of it, and in percent between phrases", () => {
    expect([0.12, 0.21, 0.28, 0.29, 0.335, 0.34, 0.44, 0.559, 0.56, 1].map(sharePhrase)).toEqual([
      "12%", "About a quarter", "About a quarter", "Almost a third", "Almost a third", "34%",
      "About half", "About half", "Most", "Most",
    ]);
  });
});

describe("voiceFacts", () => {
  it("names the neighbouring hours across midnight", () => {
    expect(lines(voiceFacts({ ...quiet, peak_hour: { hour: 0, share: 0.5 } }))).toEqual([
      "You talk most at 12 AMAbout half of everything you dictate happens between 11 and 1.",
    ]);
  });

  it("counts one of each without a plural", () => {
    expect(
      lines(voiceFacts({ ...quiet, vocabulary: 1, longest: { seconds: 4, words: 1 }, meetings_week: { count: 1, seconds: 59 } })),
    ).toEqual([
      "1 different wordAcross everything you've said so far.",
      "0:04Your longest run without stopping. It became 1 word.",
      "1 meetingcaptured this week — 1 m of talk turned into notes you can search.",
    ]);
  });

  it("leaves out what it does not know about a run or a meeting", () => {
    expect(lines(voiceFacts({ ...quiet, longest: { seconds: 70, words: 0 }, meetings_week: { count: 2, seconds: 20 } }))).toEqual([
      "1:10Your longest run without stopping.",
      "2 meetingscaptured this week, turned into notes you can search.",
    ]);
  });

  it("draws nothing, heading included, when no card has anything to say", () => {
    expect(voiceFacts(quiet)).toBe("");
  });
});
