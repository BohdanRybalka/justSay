import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { BACKEND_PORT } from "./contracts";

const REQUIRED_SOURCES: Record<string, string[]> = {
  "default-src": ["'self'"],
  "script-src": ["'self'", "'unsafe-inline'"],
  "style-src": ["'self'", "'unsafe-inline'"],
  "img-src": ["'self'", "data:"],
};

const EXACT_CONNECT_SRC = [
  "'self'",
  "ipc:",
  "http://ipc.localhost",
  `http://127.0.0.1:${BACKEND_PORT}`,
  `http://localhost:${BACKEND_PORT}`,
];

type ShippedWindow = { label?: string; dragDropEnabled?: boolean };
type ShippedConfig = { app?: { security?: { csp?: string }; windows?: ShippedWindow[] } };

function shippedConfig(): ShippedConfig {
  const configPath = fileURLToPath(new URL("../src-tauri/tauri.conf.json", import.meta.url));
  return JSON.parse(readFileSync(configPath, "utf8")) as ShippedConfig;
}

function shippedCsp(): string {
  return shippedConfig().app?.security?.csp ?? "";
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

  it("connect-src is exactly the backend loopback origins and the IPC transport", () => {
    expect(
      [...(directives["connect-src"] ?? [])].sort(),
      "connect-src must match this list exactly — a widened network allowlist is a " +
        "zero-leak regression, and both loopback origins are derived from BACKEND_PORT " +
        "so a port change cannot leave the CSP behind (ADR 028, ADR 045)",
    ).toEqual([...EXACT_CONNECT_SRC].sort());
  });
});

describe("shipped settings window (src-tauri/tauri.conf.json)", () => {
  const settingsWindow = (shippedConfig().app?.windows ?? []).find(
    (shippedWindow) => shippedWindow.label === "settings",
  );

  it("exists under the label the rest of the shell addresses it by", () => {
    expect(settingsWindow, 'no window labelled "settings" in tauri.conf.json').toBeDefined();
  });

  it("leaves drag-drop to the page instead of letting the shell intercept it", () => {
    expect(
      settingsWindow?.dragDropEnabled,
      "dragDropEnabled must be false — at Tauri's default of true the shell installs its own " +
        "drag-drop handler and the page never receives dragenter, dragover, dragleave or drop " +
        "for an external file, which kills the Transcribe drop zone silently (ADR 087)",
    ).toBe(false);
  });
});
