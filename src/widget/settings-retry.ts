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
 *
 * Four properties are contract rather than implementation detail, because the
 * schedule is worthless without them:
 *
 * - An attempt is bounded by `SETTINGS_RETRY_TIMEOUT_MS`. A backend that accepts
 *   the connection and never answers is a failed attempt like any other, not a
 *   permanently occupied slot that stops the schedule advancing.
 * - Neither `load()` nor `retryIfDue()` ever rejects, so both are safe to call
 *   unawaited — from an interval, or from the widget's own start-up path, where
 *   a rejection would abort everything sequenced after it.
 * - `load()` is an explicit request to re-read, so it restarts the schedule
 *   rather than being dropped once the first read has settled. A language
 *   changed in Settings that fails to reach the widget gets the full bounded
 *   retry, not one `console.warn`.
 * - A retry never fires while the widget is busy, because applying settings can
 *   release and re-register the global shortcut, and doing that under a held
 *   push-to-talk key loses the release event — JS-103 again. A skipped tick
 *   costs nothing: it is not a consumed attempt.
 */

import type { UserSettings } from "../api";

export type WidgetSettings = Pick<UserSettings, "language" | "shortcut">;

export const SETTINGS_RETRY_DELAYS_MS: readonly number[] = [5_000, 10_000, 20_000, 40_000, 80_000];

export const SETTINGS_RETRY_TIMEOUT_MS = 40_000;

/** Everything the retry needs from the widget, including its clock, so the
 *  schedule can be driven without waiting for it. */
export interface SettingsRetryActions {
  now(): number;
  isBusy(): boolean;
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

function delayBeforeRetry(failures: number): number | undefined {
  if (failures < 1 || failures > SETTINGS_RETRY_DELAYS_MS.length) return undefined;
  return SETTINGS_RETRY_DELAYS_MS[failures - 1];
}

function withAttemptTimeout<T>(work: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`the backend did not answer within ${SETTINGS_RETRY_TIMEOUT_MS / 1000} seconds`)),
      SETTINGS_RETRY_TIMEOUT_MS,
    );
  });
  return Promise.race([work, expiry]).finally(() => clearTimeout(timer)) as Promise<T>;
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
      const settings = await withAttemptTimeout(actions.fetchSettings());
      await actions.applySettings(settings);
      settled = true;
      shortcutApplied = true;
    } catch (e) {
      actions.reportAttemptFailed(e);
      if (!settled && !gaveUp) {
        failures += 1;
        if (delayBeforeRetry(failures) === undefined) {
          gaveUp = true;
          actions.reportGaveUp(e);
        }
      }
      if (!shortcutApplied) {
        shortcutApplied = true;
        await actions.applyFallbackShortcut().catch((fallbackError) => {
          actions.reportAttemptFailed(fallbackError);
        });
      }
    } finally {
      inFlight = false;
    }
  }

  return {
    async load(): Promise<void> {
      if (inFlight) return;
      settled = false;
      gaveUp = false;
      failures = 0;
      await attempt();
    },
    async retryIfDue(): Promise<void> {
      if (!attempted || settled || gaveUp || inFlight) return;
      if (actions.isBusy()) return;
      const delay = delayBeforeRetry(failures);
      if (delay === undefined) return;
      if (actions.now() - lastAttemptAt < delay) return;
      await attempt();
    },
  };
}
