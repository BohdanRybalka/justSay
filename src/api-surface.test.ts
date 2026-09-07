import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * knip cannot see the members of the exported `api` object.
 *
 * `api` is an object literal, and knip's `classMembers` rule covers members of
 * a class — a `canaryDeadMember` planted on the literal leaves `npx knip` at
 * exit 0 even under every `--include` rule it has. The one setting that does
 * surface them, `ignoreExportsUsedInFile: false`, reports 22 unused exported
 * types of which 20 are false positives. So the check lives here instead
 * (ADR 051).
 *
 * This reads repository files as text rather than importing them —
 * `src/tauri-config.test.ts` is the existing precedent. The limits are
 * accepted and stated: a member declared with a computed key, a spread, or a
 * runtime-built name is invisible to the extractor, and a regular-expression
 * literal carrying an unbalanced bracket would confuse the depth counter that
 * finds the end of the literal. Comments and string contents are blanked
 * before depth is counted, so a bracket inside either is harmless. The
 * extractor's own synthetic cases below pin exactly what it can see.
 */

const SRC_DIR = fileURLToPath(new URL(".", import.meta.url));
const API_FILE = resolve(SRC_DIR, "api.ts");

/**
 * The last member declared in the real `api` literal.
 *
 * A count threshold only fires on total collapse: with 25 members, a depth
 * counter that stops after the eleventh still clears it. Requiring the final
 * member is what "the extractor read the whole literal" actually means, and it
 * fails the moment truncation starts anywhere before the end.
 */
const LAST_DECLARED_MEMBER = "clearHistory";

/**
 * Members of `api` with no caller in production code, each with the reason it
 * is allowed to stay.
 *
 * `sttLocalLoad` / `sttLocalUnload`: their backend endpoints
 * `POST /stt/local/load` and `/stt/local/unload` are live. Removing only the
 * client half leaves a wired-up half with nothing driving it — the state spec
 * 105 deleted the resource-report endpoint to avoid — and removing both halves
 * is an API-surface decision nobody has taken. Listed here so the debt is
 * visible rather than ambient.
 */
const ALLOWED_WITHOUT_PRODUCTION_CALLER = ["sttLocalLoad", "sttLocalUnload"];

/**
 * `source` with comments and string contents replaced by spaces, line
 * structure preserved.
 *
 * The bracket depth that finds the end of the object literal has to be counted
 * over code only: a `)` inside a query string or a `{` inside a doc comment
 * closes the literal early and silently drops every member after it.
 */
function blankCommentsAndStrings(source: string): string {
  const out: string[] = [];
  let mode: "code" | "line" | "block" | "'" | '"' | "`" = "code";
  let i = 0;

  while (i < source.length) {
    const char = source[i];
    const next = source[i + 1];

    if (mode === "code") {
      if (char === "/" && next === "/") {
        mode = "line";
        out.push("  ");
        i += 2;
      } else if (char === "/" && next === "*") {
        mode = "block";
        out.push("  ");
        i += 2;
      } else if (char === "'" || char === '"' || char === "`") {
        mode = char;
        out.push(" ");
        i += 1;
      } else {
        out.push(char);
        i += 1;
      }
      continue;
    }

    if (mode === "line") {
      if (char === "\n") mode = "code";
      out.push(char === "\n" ? "\n" : " ");
      i += 1;
      continue;
    }

    if (mode === "block") {
      if (char === "*" && next === "/") {
        mode = "code";
        out.push("  ");
        i += 2;
        continue;
      }
      out.push(char === "\n" ? "\n" : " ");
      i += 1;
      continue;
    }

    if (char === "\\") {
      out.push("  ");
      i += 2;
      continue;
    }
    if (char === mode) mode = "code";
    out.push(char === "\n" ? "\n" : " ");
    i += 1;
  }

  return out.join("");
}

/**
 * Top-level member names of the `export const api = {` object literal.
 *
 * Both `name: value` and the method shorthand `name() { … }`, with or without
 * `async`, count as a member.
 */
function apiMemberNames(source: string): string[] {
  const code = blankCommentsAndStrings(source);
  const start = code.indexOf("export const api = {");
  if (start === -1) return [];

  const names: string[] = [];
  let depth = 0;
  for (const line of code.slice(start).split("\n").slice(1)) {
    if (depth === 0 && line.startsWith("}")) break;
    if (depth === 0) {
      const match = /^ {2}(?:async +)?([A-Za-z_$][\w$]*)\s*[:(]/.exec(line);
      if (match) names.push(match[1]);
    }
    for (const char of line) {
      if (char === "{" || char === "(" || char === "[") depth++;
      if (char === "}" || char === ")" || char === "]") depth--;
    }
  }
  return names;
}

function productionSources(): Map<string, string> {
  const sources = new Map<string, string>();
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        walk(path);
        continue;
      }
      if (!entry.endsWith(".ts") || entry.endsWith(".test.ts")) continue;
      if (resolve(path) === API_FILE) continue;
      sources.set(path, readFileSync(path, "utf8"));
    }
  };
  walk(SRC_DIR);
  return sources;
}

function membersWithoutCaller(members: string[], sources: Map<string, string>): string[] {
  return members.filter((member) => {
    const used = new RegExp(String.raw`(?<![\w$])${member}(?![\w$])`);
    return ![...sources.values()].some((source) => used.test(source));
  });
}

describe("the exported api object (src/api.ts)", () => {
  const members = apiMemberNames(readFileSync(API_FILE, "utf8"));

  it("is read to its last declared member", () => {
    expect(
      members,
      `the extractor did not reach \`${LAST_DECLARED_MEMBER}\`, the last member of ` +
        "`export const api = {`. It stopped early, so every member after the cut is " +
        "unchecked and the assertions below are silently incomplete",
    ).toContain(LAST_DECLARED_MEMBER);
  });

  it("has a production caller for every member", () => {
    const orphans = membersWithoutCaller(members, productionSources()).filter(
      (member) => !ALLOWED_WITHOUT_PRODUCTION_CALLER.includes(member),
    );

    expect(
      orphans,
      "these `api` members are called from nowhere in src/ outside api.ts and outside " +
        "tests. Delete the member and its backend endpoint together, or add it to " +
        "ALLOWED_WITHOUT_PRODUCTION_CALLER with the reason it stays (ADR 051)",
    ).toEqual([]);
  });

  it("carries no allowlist entry that has outlived its member", () => {
    const stale = ALLOWED_WITHOUT_PRODUCTION_CALLER.filter(
      (member) => !members.includes(member),
    );

    expect(
      stale,
      "these ALLOWED_WITHOUT_PRODUCTION_CALLER entries are no longer members of `api`. " +
        "A stale entry silently exempts whatever future member reuses the name",
    ).toEqual([]);
  });
});

describe("the extractor itself", () => {
  const SYNTHETIC_API = [
    "export const api = {",
    '  calledSomewhere: () => request("GET", "/a"),',
    "",
    "  /** A doc comment must not be read as a member, and neither must the",
    "   *  brace it carries: { */",
    '  bracketsInAString: () => request("GET", "/b?q=;)}"),',
    "",
    "  async shorthandMember() {",
    '    return request("GET", "/c");',
    "  },",
    "",
    '  neverCalled: () => request("GET", "/d"),',
    "};",
    "",
    "export const notApi = {",
    "  decoy: 1,",
    "};",
  ].join("\n");

  it("finds the members of the api literal and nothing after it", () => {
    expect(apiMemberNames(SYNTHETIC_API)).toEqual([
      "calledSomewhere",
      "bracketsInAString",
      "shorthandMember",
      "neverCalled",
    ]);
  });

  it("reports only the member with no caller", () => {
    const sources = new Map([
      [
        "fake-consumer.ts",
        "await api.calledSomewhere();\napi.bracketsInAString();\napi.shorthandMember();",
      ],
    ]);

    expect(membersWithoutCaller(apiMemberNames(SYNTHETIC_API), sources)).toEqual(["neverCalled"]);
  });

  /**
   * The decoy spells `calledSomewhere` verbatim, lower-case initial included:
   * the guard is a character-class boundary, not a case-insensitive one, so an
   * identifier that merely capitalises the member name would pass this test
   * with the guard broken and prove nothing.
   */
  it("does not accept an identifier that merely contains a member name as a caller", () => {
    const sources = new Map([["fake-consumer.ts", "retrycalledSomewhereOnce();"]]);

    expect(
      membersWithoutCaller(["calledSomewhere"], sources),
      "`calledSomewhere` was read as called by `retrycalledSomewhereOnce` — the " +
        "word-boundary guard is not compiling, so any member whose name is a substring " +
        "of an unrelated identifier is exempt from this whole file",
    ).toEqual(["calledSomewhere"]);
  });
});
