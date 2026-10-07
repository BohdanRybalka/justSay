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

type ShippedWindow = Record<string, unknown> & { label?: string; dragDropEnabled?: boolean };
type ShippedConfig = { app?: { security?: { csp?: string }; windows?: ShippedWindow[] } };
type Platform = "windows" | "macos";

const PLATFORMS: Platform[] = ["windows", "macos"];

function readConfig(fileName: string): ShippedConfig {
  const configPath = fileURLToPath(new URL(`../src-tauri/${fileName}`, import.meta.url));
  return JSON.parse(readFileSync(configPath, "utf8")) as ShippedConfig;
}

function shippedConfig(): ShippedConfig {
  return readConfig("tauri.conf.json");
}

function platformMainWindow(platform: Platform): ShippedWindow | undefined {
  return (readConfig(`tauri.${platform}.conf.json`).app?.windows ?? []).find(
    (shippedWindow) => shippedWindow.label === "settings",
  );
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

const PLATFORM_ONLY_KEYS: Record<Platform, Record<string, unknown>> = {
  windows: { decorations: false },
  macos: {
    decorations: true,
    titleBarStyle: "Overlay",
    hiddenTitle: true,
    trafficLightPosition: { x: 14, y: 21 },
  },
};

function withoutPlatformKeys(platform: Platform): Record<string, unknown> {
  const shared = { ...platformMainWindow(platform) };
  for (const key of Object.keys(PLATFORM_ONLY_KEYS[platform])) delete shared[key];
  return shared;
}

describe("shipped main window (src-tauri/tauri.<platform>.conf.json)", () => {
  it("is declared only per platform, because a platform file replaces the windows array whole", () => {
    expect(
      shippedConfig().app?.windows,
      "a window in the shared tauri.conf.json is dead on Windows and macOS, whose files replace " +
        "the array wholesale, and drifts from the entries that ship",
    ).toBeUndefined();
  });

  for (const platform of PLATFORMS) {
    it(`exists on ${platform} under the label the rest of the shell addresses it by`, () => {
      expect(platformMainWindow(platform), `no window labelled "settings" for ${platform}`).toBeDefined();
    });

    it(`leaves drag-drop to the page instead of letting the shell intercept it on ${platform}`, () => {
      expect(
        platformMainWindow(platform)?.dragDropEnabled,
        "dragDropEnabled must be false — at Tauri's default of true the shell installs its own " +
          "drag-drop handler and the page never receives dragenter, dragover, dragleave or drop " +
          "for an external file, which kills dropping a file on the window silently (ADR 087)",
      ).toBe(false);
    });

    it(`carries the title bar settings ${platform} needs`, () => {
      expect(platformMainWindow(platform)).toMatchObject(PLATFORM_ONLY_KEYS[platform]);
    });
  }

  it("opens as JustSay at the design's size on both platforms", () => {
    expect(withoutPlatformKeys("windows")).toMatchObject({
      title: "JustSay",
      width: 1044,
      height: 720,
      minWidth: 760,
      minHeight: 560,
      visible: false,
    });
  });

  it("agrees between platforms on everything but the title bar", () => {
    expect(withoutPlatformKeys("macos")).toEqual(withoutPlatformKeys("windows"));
  });
});
