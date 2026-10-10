// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { matchExcerpt } from "./history-entry";

describe("matchExcerpt — the text around the first match", () => {
  it("starts a few whole words before a match deep in the text, keeping the backend's markup", () => {
    const excerpt = matchExcerpt(`${"alpha ".repeat(20)}R&amp;D <mark>test</mark> &lt;b&gt;`)!;

    expect(excerpt).toMatch(/^…alpha /);
    expect(excerpt).toContain("R&amp;D <mark>test</mark> &lt;b&gt;");
    expect(excerpt.match(/alpha/g)!.length).toBeLessThan(20);
  });

  it("leaves a text alone when its first match is near the start or there is none", () => {
    expect(matchExcerpt("a <mark>test</mark> here")).toBeNull();
    expect(matchExcerpt("<mark>test</mark> here")).toBeNull();
    expect(matchExcerpt("no mark ".repeat(20))).toBeNull();
  });
});
