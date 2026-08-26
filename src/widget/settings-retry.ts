/**
 * The one decision behind re-reading the widget's settings after a failed load:
 * how often to ask again, and when to stop asking.
 *
 * It lives outside widget.ts because both halves of the answer are wrong by
 * default. Never asking again is the widget half of JS-100 — a dictation
 * language and a push-to-talk shortcut stuck at their defaults for the whole
 * session even after the backend comes up. Asking forever is what got the
 * four-line version reverted out of PR #81: the backend exempts only `/health`
 * from its token check, so a broken Tauri bridge leaves health green while
 * `/settings` answers 401 for as long as the window is open, and every poll
 * tick then paid an IPC, a refused request and a full shortcut
 * release-and-register cycle.
 *
 * ADR 047 fixes the schedule between the two: gaps doubling from the five-second
 * connection poll until one gap on its own outlasts the token-call reuse window,
 * then a single report to the user and silence. `settings-retry.test.ts` asserts
 * the last gap against `TOKEN_CALL_REUSE_MS` itself rather than a copy of it, so
 * a budget that expires inside one reuse window — and therefore never makes a
 * real second attempt — turns the suite red.
 */

import type { UserSettings } from "../api";

export type WidgetSettings = Pick<UserSettings, "language" | "shortcut">;

export const SETTINGS_RETRY_DELAYS_MS: readonly number[] = [5_000, 10_000, 20_000, 40_000, 80_000];

/** Everything the retry needs from the widget, including its clock, so the
 *  schedule can be driven without waiting for it. */
export interface SettingsRetryActions {
  now(): number;
  fetchSettings(): Promise<WidgetSettings>;
  applySettings(settings: WidgetSettings): Promise<void>;
  applyFallbackShortcut(): Promise<void>;
  reportAttemptFailed(error: unknown): void;
  reportGaveUp(error: unknown): void;
}

export interface SettingsRetry {
  load(): Promise<void>;
  retryIfDue(): Promise<void>;
}

export function createSettingsRetry(actions: SettingsRetryActions): SettingsRetry {
  let attempted = false;
  let settled = false;
  let gaveUp = false;
  let inFlight = false;
  let shortcutApplied = false;
  let failures = 0;
  let lastAttemptAt = 0;

  async function attempt(): Promise<void> {
    attempted = true;
    inFlight = true;
    lastAttemptAt = actions.now();
    try {
      const settings = await actions.fetchSettings();
      await actions.applySettings(settings);
      settled = true;
      shortcutApplied = true;
    } catch (e) {
      actions.reportAttemptFailed(e);
      if (!shortcutApplied) {
        shortcutApplied = true;
        await actions.applyFallbackShortcut();
      }
      if (settled || gaveUp) return;
      failures += 1;
      if (SETTINGS_RETRY_DELAYS_MS[failures - 1] === undefined) {
        gaveUp = true;
        actions.reportGaveUp(e);
      }
    } finally {
      inFlight = false;
    }
  }

  return {
    load(): Promise<void> {
      return attempt();
    },
    async retryIfDue(): Promise<void> {
      if (!attempted || settled || gaveUp || inFlight) return;
      if (actions.now() - lastAttemptAt < SETTINGS_RETRY_DELAYS_MS[failures - 1]) return;
      await attempt();
    },
  };
}
