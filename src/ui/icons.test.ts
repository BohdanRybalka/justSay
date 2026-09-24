// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import { icon, mountIconSprite } from "./icons";

const DESIGN_ICON_NAMES = [
  "mic", "check", "alert", "sliders", "clock", "chart", "cog", "globe", "file", "upload",
  "search", "copy", "star", "dots", "flame", "book", "sun", "moon", "chev", "x", "min", "sq",
  "stop", "refresh", "share", "folder", "cloud", "chip", "users", "speaker", "plus", "trend",
  "sparkle", "exit",
];

afterEach(() => {
  document.body.innerHTML = "";
});

describe("icon", () => {
  it("references a sprite symbol at the regular size by default", () => {
    expect(icon("mic")).toBe('<svg class="icon" aria-hidden="true"><use href="#mic"/></svg>');
  });

  it("adds a size modifier for small and large icons", () => {
    expect(icon("copy", "small")).toBe(
      '<svg class="icon icon--small" aria-hidden="true"><use href="#copy"/></svg>',
    );
    expect(icon("upload", "large")).toBe(
      '<svg class="icon icon--large" aria-hidden="true"><use href="#upload"/></svg>',
    );
  });
});

describe("mountIconSprite", () => {
  it("mounts every icon of the design once, each id unique", () => {
    mountIconSprite(document);

    const ids = [...document.querySelectorAll("symbol")].map((symbol) => symbol.id);
    expect(ids).toEqual(DESIGN_ICON_NAMES);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("keeps a single sprite when called again", () => {
    mountIconSprite(document);
    mountIconSprite(document);

    expect(document.querySelectorAll("svg#icon-sprite")).toHaveLength(1);
    expect(document.querySelectorAll("symbol#mic")).toHaveLength(1);
  });

  it("draws each symbol on the design's 24-unit grid", () => {
    mountIconSprite(document);

    const symbols = [...document.querySelectorAll("symbol")];
    expect(symbols).not.toHaveLength(0);
    for (const symbol of symbols) {
      expect(symbol.getAttribute("viewBox"), symbol.id).toBe("0 0 24 24");
      expect(symbol.children.length, symbol.id).toBeGreaterThan(0);
    }
  });
});
