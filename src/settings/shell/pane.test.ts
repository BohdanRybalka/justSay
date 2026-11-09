import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

describe("pane stylesheet", () => {
  const css = readFileSync(join(__dirname, "pane.css"), "utf-8");
  const panel = css.match(/\.panel\s*\{([^}]*)\}/)![1];

  it("keeps the panel at the design width and centres it in a wider window", () => {
    expect(panel).toMatch(/max-width:\s*830px;/);
    expect(panel).toMatch(/margin:\s*0 auto;/);
  });
});
