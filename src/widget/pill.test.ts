// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { renderMeetingIndicator } from "./meeting-indicator";
import { PILL_HOVER_CLASS, WAVE_BASE_HEIGHTS, renderPill, waveHeights, type PillView } from "./pill";

const STATE_MODIFIERS = ["pill--rest", "pill--live", "pill--alert"];

function pillFromWidgetMarkup(): HTMLElement {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  document.body.innerHTML = html
    .slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"))
    .replace(/<script[\s\S]*?<\/script>/g, "");
  return document.getElementById("widget")!;
}

function content(pill: HTMLElement): HTMLElement {
  return pill.querySelector<HTMLElement>(".pill-content")!;
}

function shownText(pill: HTMLElement): string {
  return [...content(pill).querySelectorAll(".pill-label")].map((label) => label.textContent).join(" ");
}

function modifiers(pill: HTMLElement): string[] {
  return STATE_MODIFIERS.filter((modifier) => pill.classList.contains(modifier));
}

function glyph(pill: HTMLElement): string | null | undefined {
  return content(pill).querySelector("use")?.getAttribute("href");
}

function barHeights(pill: HTMLElement): string[] {
  return [...content(pill).querySelectorAll<HTMLElement>(".pill-wave i")].map((bar) => bar.style.height);
}

const EVERY_VIEW: PillView[] = [
  { kind: "rest", hint: "Ctrl + Alt + V" },
  { kind: "listening", elapsedSeconds: 7, level: 0.5 },
  { kind: "working" },
  { kind: "done", words: 117 },
  { kind: "noSpeech" },
  { kind: "alert", label: "No connection" },
];

describe("the pill at rest", () => {
  it("is the microphone with the shortcut hint beside it", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "rest", hint: "Ctrl + Alt + V" });

    expect(modifiers(pill)).toEqual(["pill--rest"]);
    expect(glyph(pill)).toBe("#mic");
    expect(content(pill).querySelector(".pill-hint")?.textContent).toBe("Ctrl + Alt + V");
  });

  it("writes the hint as text, because a shortcut comes from the user's settings", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "rest", hint: "<b>Ctrl</b>" });

    expect(pill.querySelector(".pill-hint b")).toBeNull();
    expect(pill.querySelector(".pill-hint")?.textContent).toBe("<b>Ctrl</b>");
  });
});

describe("the pill while something happens", () => {
  it.each([
    [{ kind: "listening", elapsedSeconds: 7.9, level: 0 }, ["pill--live"], null, "0:07"],
    [{ kind: "working" }, [], null, "Writing it down"],
    [{ kind: "done", words: 117 }, [], "#check", "117 words · copied"],
    [{ kind: "done", words: 1 }, [], "#check", "1 word · copied"],
    [{ kind: "noSpeech" }, [], null, "No speech"],
    [{ kind: "alert", label: "Mic is busy" }, ["pill--alert"], "#alert", "Mic is busy"],
  ] as [PillView, string[], string | null, string][])(
    "shows %o with its own modifier, glyph and text",
    (view, expectedModifiers, expectedGlyph, expectedText) => {
      const pill = pillFromWidgetMarkup();
      renderPill(pill, { kind: "rest", hint: "Ctrl + Alt + V" });

      renderPill(pill, view);

      expect(modifiers(pill)).toEqual(expectedModifiers);
      expect(glyph(pill) ?? null).toBe(expectedGlyph);
      expect(shownText(pill)).toBe(expectedText);
    },
  );

  it("draws the listening state as the dot, seven bars and a clock in tabular figures", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "listening", elapsedSeconds: 64, level: 0 });

    expect(content(pill).querySelector(".pill-dot")).not.toBeNull();
    expect(barHeights(pill)).toHaveLength(7);
    expect(content(pill).querySelector(".pill-readout")?.textContent).toBe("1:04");
    expect(content(pill).querySelector(".pill-readout")?.classList.contains("num")).toBe(true);
  });

  it("draws the working state as three dots", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "working" });

    expect(content(pill).querySelectorAll(".pill-dots i")).toHaveLength(3);
  });
});

describe("the wave", () => {
  it("is the design's shape at full voice and flat at silence", () => {
    expect(waveHeights(1)).toEqual([...WAVE_BASE_HEIGHTS]);
    expect(new Set(waveHeights(0)).size).toBe(1);
    expect(Math.max(...waveHeights(0))).toBeLessThan(Math.min(...WAVE_BASE_HEIGHTS));
  });

  it("grows every bar with the level and never past the design's shape", () => {
    const quiet = waveHeights(0.2);
    const loud = waveHeights(0.8);

    quiet.forEach((height, index) => expect(loud[index]).toBeGreaterThan(height));
    expect(waveHeights(3)).toEqual([...WAVE_BASE_HEIGHTS]);
    expect(waveHeights(-1)).toEqual(waveHeights(0));
  });

  it("sets each bar to the height the level asks for", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "listening", elapsedSeconds: 1, level: 0.5 });

    expect(barHeights(pill)).toEqual(waveHeights(0.5).map((height) => `${height}px`));
  });

  it("moves the same bars on a level update, so the dot keeps blinking", () => {
    const pill = pillFromWidgetMarkup();
    renderPill(pill, { kind: "listening", elapsedSeconds: 1, level: 0 });
    const dot = content(pill).querySelector(".pill-dot");
    const firstBar = content(pill).querySelector(".pill-wave i");

    renderPill(pill, { kind: "listening", elapsedSeconds: 2, level: 1 });

    expect(content(pill).querySelector(".pill-dot")).toBe(dot);
    expect(content(pill).querySelector(".pill-wave i")).toBe(firstBar);
    expect(barHeights(pill)).toEqual(WAVE_BASE_HEIGHTS.map((height) => `${height}px`));
    expect(content(pill).querySelector(".pill-readout")?.textContent).toBe("0:02");
  });
});

describe("what a repaint leaves alone", () => {
  it("keeps the hover look, which only the shell's cursor loop sets", () => {
    const pill = pillFromWidgetMarkup();
    pill.classList.add(PILL_HOVER_CLASS);

    for (const view of EVERY_VIEW) {
      renderPill(pill, view);
      expect(pill.classList.contains(PILL_HOVER_CLASS)).toBe(true);
    }
  });

  it("keeps the meeting clock's own readout, so a running meeting never blanks", () => {
    const pill = pillFromWidgetMarkup();
    renderMeetingIndicator(pill, { active: true, elapsedSeconds: 724, incident: null, mic: 0, system: 0 });
    const meetingClock = pill.querySelector(".pill-meeting .pill-readout")!;

    for (const view of EVERY_VIEW) {
      renderPill(pill, view);
      expect(pill.querySelector(".pill-meeting .pill-readout")).toBe(meetingClock);
      expect(meetingClock.textContent).toBe("12:04");
    }
  });
});
