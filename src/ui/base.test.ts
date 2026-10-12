import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

describe("base stylesheet", () => {
  const css = readFileSync(join(__dirname, "base.css"), "utf-8");
  const reducedMotion = css.slice(css.indexOf("@media (prefers-reduced-motion:reduce)"));

  it("applies the reduced-motion block to everything, pseudo-elements included", () => {
    expect(reducedMotion).toContain("*,*::before,*::after{");
  });

  it("ends every animation after one near-instant run, which keeps the looping ones from strobing", () => {
    expect(reducedMotion).toContain("animation-duration:.001ms!important");
    expect(reducedMotion).toContain("animation-iteration-count:1!important");
    expect(reducedMotion).toContain("transition-duration:.001ms!important");
  });
});
