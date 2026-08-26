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
 * Five properties are contract rather than implementation detail, because the
 * schedule is worthless without them:
 *
 * - `SETTINGS_RETRY_TIMEOUT_MS` bounds a whole attempt — the fetch and the apply
 *   together. The apply is the half that reaches the Tauri bridge, which can
 *   hang with no reject channel (ADR 028), so bounding the fetch alone leaves an
 *   attempt that never ends and a schedule that never advances again.
 * - Neither `load()` nor `retryIfDue()` ever rejects, so both are safe to call
 *   unawaited — from an interval, or from the widget's own start-up path, where
 *   awaiting a settings read would keep the window hidden until it answered.
 * - `load()` is an explicit request to re-read, so it restarts the schedule
 *   rather than being dropped once the first read has settled. A request that
 *   arrives while an attempt is running is remembered and served when that
 *   attempt ends: dropping it loses a language the user just chose, with no
 *   backend failure anywhere.
 * - A retry never fires while the widget is busy, because applying settings can
 *   release and re-register the global shortcut, and doing that under a held
 *   push-to-talk key loses the release event — JS-103 again. The check is made
 *   after the fetch answers as well as before it starts, since the widget can
 *   become busy while the request is out. Settings fetched into a busy widget
 *   are held and applied on a later tick; holding them is not a failed attempt,
 *   so it consumes no retry and cannot exhaust the schedule.
 * - A gap is satisfied within half a poll period of its nominal length. Every
 *   gap is a multiple of the poll that drives it, so an exact comparison lands
 *   on the tick boundary and slips a whole period whenever a tick arrives a hair
 *   early.
 */

import type { UserSettings } from "../api";
import { withTimeout } from "../timeout";

export type WidgetSettings = Pick<UserSettings, "language" | "shortcut">;

export const CONNECTION_POLL_MS = 5_000;

export const SETTINGS_RETRY_DELAYS_MS: readonly number[] = [5_000, 10_000, 20_000, 40_000, 80_000];

export const SETTINGS_RETRY_TIMEOUT_MS = 40_000;

export const SETTINGS_RETRY_DUE_TOLERANCE_MS = CONNECTION_POLL_MS / 2;

/** Everything the retry needs from the widget, including its clock, so the
 *  schedule can be driven without waiting for it. */
export interface SettingsRetryActions {
  now(): number;
  isBusy(): boolean;
  fetchSettings(): Promise<WidgetSettings>;
  applySettings(settings: WidgetSettings): Promise<void>;
  applyFallbackShortcut(): Promise<void>;
  reportAttemptFailed(error: unknown): void;
  reportFallbackFailed(error: unknown): void;
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

export function createSettingsRetry(actions: SettingsRetryActions): SettingsRetry {
  let attempted = false;
  let settled = false;
  let gaveUp = false;
  let inFlight = false;
  let shortcutApplied = false;
  let reloadRequested = false;
  let held: WidgetSettings | null = null;
  let failures = 0;
  let lastAttemptAt = 0;

  async function fetchAndApply(): Promise<void> {
    const settings = held ?? (await actions.fetchSettings());
    if (actions.isBusy()) {
      held = settings;
      return;
    }
    held = null;
    await actions.applySettings(settings);
    settled = true;
    shortcutApplied = true;
  }

  async function attempt(): Promise<void> {
    attempted = true;
    inFlight = true;
    try {
      await withTimeout(fetchAndApply(), SETTINGS_RETRY_TIMEOUT_MS);
    } catch (e) {
      held = null;
      actions.reportAttemptFailed(e);
      failures += 1;
      if (delayBeforeRetry(failures) === undefined) {
        gaveUp = true;
        actions.reportGaveUp(e);
      }
      if (!shortcutApplied) {
        shortcutApplied = true;
        await actions.applyFallbackShortcut().catch((fallbackError) => {
          actions.reportFallbackFailed(fallbackError);
        });
      }
    } finally {
      lastAttemptAt = actions.now();
      inFlight = false;
    }
  }

  function restartSchedule(): void {
    settled = false;
    gaveUp = false;
    failures = 0;
    held = null;
  }

  async function attemptThenServePendingReload(): Promise<void> {
    await attempt();
    while (reloadRequested) {
      reloadRequested = false;
      restartSchedule();
      await attempt();
    }
  }

  return {
    async load(): Promise<void> {
      if (inFlight) {
        reloadRequested = true;
        return;
      }
      restartSchedule();
      await attemptThenServePendingReload();
    },
    async retryIfDue(): Promise<void> {
      if (!attempted || settled || gaveUp || inFlight) return;
      if (actions.isBusy()) return;
      if (held === null) {
        const delay = delayBeforeRetry(failures);
        if (delay === undefined) return;
        if (actions.now() - lastAttemptAt < delay - SETTINGS_RETRY_DUE_TOLERANCE_MS) return;
      }
      await attemptThenServePendingReload();
    },
  };
}
