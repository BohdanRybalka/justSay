import { readFileSync, readdirSync } from "fs";
import { join, relative, resolve } from "path";
import { describe, expect, it } from "vitest";

const REPO_ROOT = resolve(__dirname, "../..");
const TOKENS_PATH = "src/ui/tokens.css";
const STYLESHEET_WITH_OWN_COLOURS = "src/widget/widget.css";

function readRepoFile(path: string): string {
  return readFileSync(join(REPO_ROOT, path), "utf-8");
}

function tokenNames(block: string): string[] {
  return [...block.matchAll(/(--[\w-]+)\s*:/g)].map((match) => match[1]);
}

function themeBlock(css: string, selector: string): string {
  const start = css.indexOf(`${selector}{`);
  expect(start, `${selector} block in ${TOKENS_PATH}`).toBeGreaterThanOrEqual(0);
  return css.slice(start, css.indexOf("}", start));
}

function stylesheetsUnderSrc(dir = join(REPO_ROOT, "src")): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return stylesheetsUnderSrc(path);
    return entry.name.endsWith(".css") ? [relative(REPO_ROOT, path).replace(/\\/g, "/")] : [];
  });
}

const COLOUR_LITERAL = /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\([^)]*\)/gi;
const WHITE = /^(?:#fff|#ffffff|rgba?\(\s*255\s*,\s*255\s*,\s*255\s*(?:,\s*[\d.]+\s*)?\))$/i;

function colourDeclarations(path: string): string[] {
  const css = readRepoFile(path).replace(/\/\*[\s\S]*?\*\//g, "");
  return [...css.matchAll(/([\w-]+)\s*:\s*([^;{}]+)[;}]/g)]
    .filter(([, , value]) => (value.match(COLOUR_LITERAL) ?? []).some((colour) => !WHITE.test(colour)))
    .map(([, property, value]) => `${path}: ${property}: ${value.trim()}`);
}

describe("design tokens", () => {
  it("defines every token in both the light and the dark theme", () => {
    const css = readRepoFile(TOKENS_PATH);
    const light = tokenNames(themeBlock(css, ":root"));
    const dark = tokenNames(themeBlock(css, '[data-theme="dark"]'));

    expect(light).toContain("--canvas");
    expect(light).toContain("--orange");
    expect([...dark].sort()).toEqual([...light].sort());
  });
});

describe("colours outside the tokens", () => {
  const checked = stylesheetsUnderSrc().filter(
    (path) => path !== TOKENS_PATH && path !== STYLESHEET_WITH_OWN_COLOURS,
  );

  it("walks the app's stylesheets", () => {
    expect(checked).toContain("src/settings/settings.css");
    expect(checked).toContain("src/ui/base.css");
  });

  it("appear in no stylesheet except as white ink on colour", () => {
    expect(checked.flatMap(colourDeclarations)).toEqual([]);
  });

  it("are still exempt only in the widget stylesheet that keeps its own palette", () => {
    expect(colourDeclarations(STYLESHEET_WITH_OWN_COLOURS)).not.toEqual([]);
  });
});
