/**
 * The one answer to "the backend has not answered yet - is that a failure?"
 * (ADR 092).
 *
 * A window that has had no answer from `/health` is starting; a window whose
 * request an answering backend failed has failed. `BACKEND_WAIT_BUDGET_MS`
 * bounds both how long one settings load may hang and how long this decision
 * will call a window starting, so the two numbers cannot drift apart.
 */

export const BACKEND_WAIT_BUDGET_MS = 40_000;

export type BackendStartupScreen = "ready" | "starting" | "failed";

/** What a window knows about its own start-up at the moment it asks.
 *
 *  `msWaiting` counts from the moment the window began waiting and must come
 *  from a monotonic clock, never from `Date.now()`. `answeredFailures` counts
 *  only loads a backend answered and refused. */
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

/** Whether a window has waited longer than any window may call itself
 *  starting. The shared fact behind every waiting screen and label. */
export function hasOutlastedStartupBudget(msWaiting: number): boolean {
  return msWaiting >= BACKEND_WAIT_BUDGET_MS;
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
  return hasOutlastedStartupBudget(state.msWaiting) ? "failed" : "starting";
}
