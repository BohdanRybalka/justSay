import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

describe("pane stylesheet", () => {
  const css = readFileSync(join(__dirname, "pane.css"), "utf-8");
  const panel = css.match(/\.panel\s*\{([^}]*)\}/)![1];

  it("caps the panel at the design width with automatic side margins", () => {
    expect(panel).toMatch(/max-width:\s*830px;/);
    expect(panel).toMatch(/margin:\s*0 auto;/);
  });
});
