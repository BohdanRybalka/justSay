import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// The shipped Content-Security-Policy is enforced verbatim by the WebView:
// Tauri does not repair it at runtime (ADR 028). It has no build-time
// validation, and a missing source fails silently inside the WebView — on
// macOS it produced a green backend badge on a completely inert window.
// This test is the only place a regression in that string can be caught, so
// it asserts every source the Tauri bridge needs, not only the ones our own
// code uses. Each entry's reason is in ADR 028; do not drop one because the
// app's own traffic does not appear to need it.
const REQUIRED_SOURCES: Record<string, string[]> = {
  // Baseline for every directive not stated below.
  "default-src": ["'self'"],
  // Stated explicitly: `wry` injects the bridge (`__TAURI_INTERNALS__`,
  // `window.ipc`) as a platform init script, which is not a same-origin
  // resource, so inheriting `'self'` from default-src blocks it.
  "script-src": ["'self'", "'unsafe-inline'"],
  // `ipc:` / `http://ipc.localhost` are the IPC transport's own origins
  // (`ipc://localhost/<cmd>` on macOS, `http://ipc.localhost/<cmd>` on
  // Windows); the loopback pair is the Python backend.
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
