/**
 * The one answer to "the backend has not answered yet - is that a failure?"
 * (ADR 092).
 *
 * A window that has had no answer from `/health` is starting; a window whose
 * request an answering backend failed has failed. `BACKEND_WAIT_BUDGET_MS`
 * bounds both how long one settings load may hang and how long a window may
 * keep saying it is starting, so the two numbers cannot drift apart.
 */

export const BACKEND_WAIT_BUDGET_MS = 40_000;

export type BackendStartupScreen = "ready" | "starting" | "failed";

/** What a window knows about its own start-up at the moment it asks.
 *
 *  `msWaiting` is measured from the window's first `/health` attempt, and
 *  `answeredFailures` counts only loads that failed against a backend that was
 *  answering. */
export interface BackendStartupState {
  loaded: boolean;
  loadInFlight: boolean;
  backendAnswering: boolean;
  answeredFailures: number;
  msWaiting: number;
}

/** The screen the window shows, and whether it starts a load now. */
export interface BackendStartupDecision {
  screen: BackendStartupScreen;
  load: boolean;
}

/** Total and pure: no clock, no timer and no request.
 *
 *  `load` stays true past `BACKEND_WAIT_BUDGET_MS`, so a window that has given
 *  up saying it is starting still recovers the moment `/health` answers. */
export function nextBackendStartup(state: BackendStartupState): BackendStartupDecision {
  return {
    screen: startupScreen(state),
    load:
      !state.loaded &&
      !state.loadInFlight &&
      state.backendAnswering &&
      state.answeredFailures === 0,
  };
}

function startupScreen(state: BackendStartupState): BackendStartupScreen {
  if (state.loaded) return "ready";
  if (state.answeredFailures > 0) return "failed";
  return state.msWaiting >= BACKEND_WAIT_BUDGET_MS ? "failed" : "starting";
}
