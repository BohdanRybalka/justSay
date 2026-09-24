// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { applyThemePreference } from "./theme";

interface FakeColourScheme {
  matches: boolean;
  addEventListener: (type: "change", listener: () => void) => void;
  removeEventListener: (type: "change", listener: () => void) => void;
  switchTo: (dark: boolean) => void;
}

function stubColourScheme(dark: boolean): FakeColourScheme {
  const listeners = new Set<() => void>();
  const scheme: FakeColourScheme = {
    matches: dark,
    addEventListener: (_type, listener) => listeners.add(listener),
    removeEventListener: (_type, listener) => listeners.delete(listener),
    switchTo: (nowDark) => {
      scheme.matches = nowDark;
      listeners.forEach((listener) => listener());
    },
  };
  vi.stubGlobal("matchMedia", (query: string) => {
    expect(query).toBe("(prefers-color-scheme: dark)");
    return scheme;
  });
  return scheme;
}

function currentTheme(): string | undefined {
  return document.documentElement.dataset.theme;
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete document.documentElement.dataset.theme;
});

describe("applyThemePreference", () => {
  it("takes the OS colour scheme when following the system", () => {
    stubColourScheme(true);

    applyThemePreference("system");

    expect(currentTheme()).toBe("dark");
  });

  it("follows the OS when its colour scheme changes while the page is open", () => {
    const scheme = stubColourScheme(false);
    applyThemePreference("system");
    expect(currentTheme()).toBe("light");

    scheme.switchTo(true);
    expect(currentTheme()).toBe("dark");

    scheme.switchTo(false);
    expect(currentTheme()).toBe("light");
  });

  it("stops following the OS once a fixed theme is chosen", () => {
    const scheme = stubColourScheme(false);
    applyThemePreference("system");

    applyThemePreference("light");
    scheme.switchTo(true);

    expect(currentTheme()).toBe("light");
  });
});
