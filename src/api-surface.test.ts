import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
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
 * runtime-built name is invisible to the extractor, and a member name that is
 * a substring of an unrelated identifier would read as used. The extractor's
 * own synthetic case below pins exactly what it can see.
 */

const SRC_DIR = fileURLToPath(new URL(".", import.meta.url));
const API_FILE = "api.ts";

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

/** Top-level member names of the `export const api = {` object literal. */
function apiMemberNames(source: string): string[] {
  const start = source.indexOf("export const api = {");
  if (start === -1) return [];

  const names: string[] = [];
  let depth = 0;
  for (const line of source.slice(start).split("\n").slice(1)) {
    if (depth === 0 && line.startsWith("}")) break;
    if (depth === 0) {
      const match = /^ {2}([A-Za-z_$][\w$]*)\s*:/.exec(line);
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
      if (!entry.endsWith(".ts") || entry.endsWith(".test.ts") || entry === API_FILE) continue;
      sources.set(path, readFileSync(path, "utf8"));
    }
  };
  walk(SRC_DIR);
  return sources;
}

function membersWithoutCaller(members: string[], sources: Map<string, string>): string[] {
  return members.filter((member) => {
    const used = new RegExp(`(?<![\w$])${member}(?![\w$])`);
    return ![...sources.values()].some((source) => used.test(source));
  });
}

describe("the exported api object (src/api.ts)", () => {
  const members = apiMemberNames(readFileSync(join(SRC_DIR, API_FILE), "utf8"));

  it("declares members the extractor can actually see", () => {
    expect(
      members.length,
      "the extractor found no members of `export const api = {` — it has stopped " +
        "matching the object literal, and every assertion below is now vacuously true",
    ).toBeGreaterThan(10);
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
    "  /** A doc comment must not be read as a member. */",
    '  neverCalled: () => request("GET", "/b"),',
    "};",
    "",
    "export const notApi = {",
    "  decoy: 1,",
    "};",
  ].join("\n");

  it("finds the members of the api literal and nothing after it", () => {
    expect(apiMemberNames(SYNTHETIC_API)).toEqual(["calledSomewhere", "neverCalled"]);
  });

  it("reports only the member with no caller", () => {
    const sources = new Map([["fake-consumer.ts", "await api.calledSomewhere();"]]);

    expect(membersWithoutCaller(apiMemberNames(SYNTHETIC_API), sources)).toEqual(["neverCalled"]);
  });
});
