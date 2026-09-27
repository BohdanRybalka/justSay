/**
 * Light or dark palette for a page: sets `data-theme` on `<html>`, which the
 * tokens in `tokens.css` key on. `system` follows the OS setting while the
 * page is open, not only at load.
 */

import { setTheme } from "@tauri-apps/api/app";

export type ThemePreference = "system" | "light" | "dark";

const DARK_SCHEME_QUERY = "(prefers-color-scheme: dark)";

let stopFollowingSystem = (): void => {};

export function applyThemePreference(preference: ThemePreference): void {
  const root = document.documentElement;
  stopFollowingSystem();
  stopFollowingSystem = () => {};
  if (preference !== "system") {
    root.dataset.theme = preference;
    return;
  }
  const scheme = window.matchMedia(DARK_SCHEME_QUERY);
  const followScheme = (): void => {
    root.dataset.theme = scheme.matches ? "dark" : "light";
  };
  followScheme();
  scheme.addEventListener("change", followScheme);
  stopFollowingSystem = () => scheme.removeEventListener("change", followScheme);
}

/** This page's theme, and the same choice for the app's native parts —
 *  scrollbars, the macOS title area — which also set every page's
 *  `prefers-color-scheme`. A page served outside the app has no native parts
 *  and keeps its own theme alone. */
export async function applyAppTheme(preference: ThemePreference): Promise<void> {
  applyThemePreference(preference);
  try {
    await setTheme(preference === "system" ? null : preference);
  } catch {
  }
}
