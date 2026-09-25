// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { physicalPillRect, watchPillRect, type PillRect } from "./pill-rect";

interface FakeResolutionQuery {
  query: string;
  change: () => void;
}

function stubResizeObserver(): { resize: () => void } {
  let callback: () => void = () => {};
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(onResize: () => void) {
        callback = onResize;
      }
      observe() {
        callback();
      }
    },
  );
  return { resize: () => callback() };
}

function stubResolutionQueries(): FakeResolutionQuery[] {
  const queries: FakeResolutionQuery[] = [];
  vi.stubGlobal("matchMedia", (query: string) => ({
    addEventListener: (_type: "change", listener: () => void) =>
      queries.push({ query, change: listener }),
  }));
  return queries;
}

function pillAt(left: number, top: number, width: number, height: number): Element {
  const pill = document.createElement("div");
  pill.getBoundingClientRect = () => new DOMRect(left, top, width, height);
  return pill;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("physicalPillRect", () => {
  it("scales the CSS box by the pixel ratio and rounds to whole pixels", () => {
    expect(physicalPillRect(new DOMRect(40, 12, 160, 40), 1.25)).toEqual({
      x: 50,
      y: 15,
      width: 200,
      height: 50,
    });
  });
});

describe("watchPillRect", () => {
  it("reports the pill at once and again when its size changes", () => {
    const observer = stubResizeObserver();
    stubResolutionQueries();
    vi.stubGlobal("devicePixelRatio", 2);
    const reports: PillRect[] = [];

    watchPillRect(pillAt(40, 12, 160, 40), (rect) => reports.push(rect));
    observer.resize();

    expect(reports).toEqual([
      { x: 80, y: 24, width: 320, height: 80 },
      { x: 80, y: 24, width: 320, height: 80 },
    ]);
  });

  it("reports again in the new ratio after each pixel-ratio change", () => {
    stubResizeObserver();
    const queries = stubResolutionQueries();
    vi.stubGlobal("devicePixelRatio", 1);
    const reports: PillRect[] = [];
    watchPillRect(pillAt(40, 12, 160, 40), (rect) => reports.push(rect));
    expect(queries.map((q) => q.query)).toEqual(["(resolution: 1dppx)"]);

    vi.stubGlobal("devicePixelRatio", 1.5);
    queries[0].change();
    vi.stubGlobal("devicePixelRatio", 2);
    queries[1].change();

    expect(reports.map((rect) => rect.width)).toEqual([160, 240, 320]);
    expect(queries.map((q) => q.query)).toEqual([
      "(resolution: 1dppx)",
      "(resolution: 1.5dppx)",
      "(resolution: 2dppx)",
    ]);
  });
});
