import { describe, expect, it } from "vitest";
import { drawSavedCard, layoutSavedCard, wrapLines, type SavedCardView } from "./insights-image";

const CHAR_WIDTH = 7;

function recordingContext() {
  const drawn: { text: string; font: string; x: number; y: number; alpha: number }[] = [];
  const gradient = { addColorStop: () => {} };
  const ctx = {
    font: "",
    letterSpacing: "0px",
    globalAlpha: 1,
    fillStyle: "" as unknown,
    textBaseline: "alphabetic",
    measureText: (text: string) => ({ width: text.length * CHAR_WIDTH }),
    fillText(text: string, x: number, y: number) {
      drawn.push({ text, font: ctx.font, x, y, alpha: ctx.globalAlpha });
    },
    createLinearGradient: () => gradient,
    save() {},
    restore() {},
    beginPath() {},
    roundRect() {},
    clip() {},
    fillRect() {},
    arc() {},
    fill() {},
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, drawn };
}

const VIEW: SavedCardView = {
  value: "2 h 57 m",
  note: "That's what typing these words by hand would have cost you. Roughly one free afternoon.",
  compare: [
    { label: "Typing", fraction: 1, time: "4:27", dim: true },
    { label: "Speaking", fraction: 0.34, time: "1:30", dim: false },
  ],
  figures: [
    { value: "10,666", label: "words in 116 recordings" },
    { value: "8 days", label: "streak · your longest" },
    { value: "118 wpm", label: "three times your typing" },
  ],
};

describe("the card drawn as an image", () => {
  it("draws every line of the card, top to bottom, in the card's fonts", () => {
    const { ctx, drawn } = recordingContext();

    drawSavedCard(ctx, VIEW, layoutSavedCard(ctx, VIEW));

    expect(drawn.map((run) => run.text)).toEqual([
      "YOU SAVED THIS MONTH",
      "2 h 57 m",
      "That's what typing these words by hand would",
      "have cost you. Roughly one free afternoon.",
      "Typing",
      "4:27",
      "Speaking",
      "1:30",
      "10,666",
      "words in 116 recordings",
      "8 days",
      "streak · your longest",
      "118 wpm",
      "three times your typing",
    ]);
    expect(drawn[1].font).toBe(`740 56px "JetBrains Mono", ui-monospace, monospace`);
    expect(drawn[2].alpha).toBe(0.9);
    const rows = drawn.map((run) => run.y);
    expect(rows.slice(0, 5)).toEqual([...rows.slice(0, 5)].sort((a, b) => a - b));
  });

  it("sets each figure beside the last, spaced by the wider of its two lines", () => {
    const { ctx, drawn } = recordingContext();

    drawSavedCard(ctx, VIEW, layoutSavedCard(ctx, VIEW));

    const starts = ["10,666", "8 days", "118 wpm"].map((value) => drawn.find((run) => run.text === value)!.x);
    expect(starts).toEqual([32, 32 + 23 * CHAR_WIDTH + 38, 32 + 23 * CHAR_WIDTH + 38 + 21 * CHAR_WIDTH + 38]);
  });

  it("is shorter by the bars when the card has none", () => {
    const { ctx, drawn } = recordingContext();

    const withBars = layoutSavedCard(ctx, VIEW).height;
    const without = layoutSavedCard(ctx, { ...VIEW, compare: [] });
    drawSavedCard(ctx, { ...VIEW, compare: [] }, without);

    expect(withBars - without.height).toBe(22 + 2 * (9 + 18));
    expect(drawn.some((run) => run.text === "Typing")).toBe(false);
  });
});

describe("wrapping", () => {
  it("breaks between words at the width and keeps a long word whole", () => {
    const { ctx } = recordingContext();

    expect(wrapLines(ctx, "one two three", 7 * CHAR_WIDTH)).toEqual(["one two", "three"]);
    expect(wrapLines(ctx, "extraordinarily long", 5 * CHAR_WIDTH)).toEqual(["extraordinarily", "long"]);
  });
});
