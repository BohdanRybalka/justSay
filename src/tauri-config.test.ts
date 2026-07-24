import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const REQUIRED_SOURCES: Record<string, string[]> = {
  "default-src": ["'self'"],
  "script-src": ["'self'", "'unsafe-inline'"],
  "connect-src": [
    "'self'",
    "ipc:",
    "http://ipc.localhost",
    "http://127.0.0.1:9377",
    "http://localhost:9377",
  ],
  "style-src": ["'self'", "'unsafe-inline'"],
  "img-src": ["'self'", "data:"],
};

function shippedCsp(): string {
  const configPath = fileURLToPath(new URL("../src-tauri/tauri.conf.json", import.meta.url));
  const config = JSON.parse(readFileSync(configPath, "utf8"));
  return config.app?.security?.csp ?? "";
}

function parseCsp(csp: string): Record<string, string[]> {
  const directives: Record<string, string[]> = {};
  for (const part of csp.split(";")) {
    const tokens = part.trim().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) continue;
    directives[tokens[0]] = tokens.slice(1);
  }
  return directives;
}

describe("shipped CSP (src-tauri/tauri.conf.json)", () => {
  const directives = parseCsp(shippedCsp());

  for (const [directive, sources] of Object.entries(REQUIRED_SOURCES)) {
    it(`${directive} lists ${sources.join(" ")}`, () => {
      expect(
        Object.keys(directives),
        `CSP directive "${directive}" is missing entirely — see ADR 028`,
      ).toContain(directive);

      for (const source of sources) {
        expect(
          directives[directive],
          `CSP directive "${directive}" is missing the source "${source}" — see ADR 028`,
        ).toContain(source);
      }
    });
  }
});
