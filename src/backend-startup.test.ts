import { describe, expect, it } from "vitest";
import {
  BACKEND_WAIT_BUDGET_MS,
  hasOutlastedStartupBudget,
  nextBackendStartup,
  type BackendStartupState,
} from "./backend-startup";

/** A window that has just come up: nothing loaded, nothing in flight, no answer
 *  from `/health` yet and no attempt behind it. */
function coldStart(overrides: Partial<BackendStartupState> = {}): BackendStartupState {
  return {
    loaded: false,
    loadInFlight: false,
    backendAnswering: false,
    answeredFailures: 0,
    msWaiting: 0,
    ...overrides,
  };
}

describe("the screen a window waiting for its backend shows", () => {
  it("is still starting one millisecond before the wait budget runs out", () => {
    const decision = nextBackendStartup(coldStart({ msWaiting: BACKEND_WAIT_BUDGET_MS - 1 }));

    expect(decision.screen).toBe("starting");
  });

  it("is failed on the millisecond the wait budget runs out", () => {
    const decision = nextBackendStartup(coldStart({ msWaiting: BACKEND_WAIT_BUDGET_MS }));

    expect(decision.screen).toBe("failed");
  });

  it("is ready once settings are held, however long the wait took", () => {
    const decision = nextBackendStartup(
      coldStart({ loaded: true, msWaiting: BACKEND_WAIT_BUDGET_MS * 10 }),
    );

    expect(decision.screen).toBe("ready");
  });

  it("is failed as soon as one attempt failed against an answering backend", () => {
    const decision = nextBackendStartup(
      coldStart({ backendAnswering: true, answeredFailures: 1, msWaiting: 0 }),
    );

    expect(decision.screen).toBe("failed");
    expect(decision.load).toBe(false);
  });
});

describe("the budget every waiting surface shares", () => {
  it("has not run out one millisecond before it does", () => {
    expect(hasOutlastedStartupBudget(BACKEND_WAIT_BUDGET_MS - 1)).toBe(false);
  });

  it("has run out on the millisecond it does", () => {
    expect(hasOutlastedStartupBudget(BACKEND_WAIT_BUDGET_MS)).toBe(true);
  });

  it("is the same fact the screen decision turns on, on both sides of the boundary", () => {
    for (const msWaiting of [BACKEND_WAIT_BUDGET_MS - 1, BACKEND_WAIT_BUDGET_MS]) {
      expect(nextBackendStartup(coldStart({ msWaiting })).screen === "failed").toBe(
        hasOutlastedStartupBudget(msWaiting),
      );
    }
  });
});

describe("whether a waiting window starts a settings load", () => {
  it("loads on the first state where /health answers and nothing else is running", () => {
    const decision = nextBackendStartup(coldStart({ backendAnswering: true }));

    expect(decision.load).toBe(true);
  });

  it("does not load on the otherwise identical state where /health is not answering", () => {
    const decision = nextBackendStartup(coldStart({ backendAnswering: false }));

    expect(decision.load).toBe(false);
  });

  it("does not load while a load is already in flight", () => {
    const decision = nextBackendStartup(coldStart({ backendAnswering: true, loadInFlight: true }));

    expect(decision.load).toBe(false);
  });

  it("does not load again once settings are held", () => {
    const decision = nextBackendStartup(coldStart({ backendAnswering: true, loaded: true }));

    expect(decision.load).toBe(false);
  });

  it("still loads after the wait budget has run out, so the failure screen heals itself", () => {
    const decision = nextBackendStartup(
      coldStart({ backendAnswering: true, msWaiting: BACKEND_WAIT_BUDGET_MS + 1 }),
    );

    expect(decision.screen).toBe("failed");
    expect(decision.load).toBe(true);
  });
});
