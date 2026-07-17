import { describe, expect, it } from "vitest";
import { computeIndicatorState, isActivationKey, onIndicatorStateChange } from "./status-indicator";

describe("computeIndicatorState", () => {
  it("active: false yields idle regardless of ready/error", () => {
    expect(computeIndicatorState({ active: false, ready: false, error: null })).toBe("idle");
    expect(computeIndicatorState({ active: false, ready: true, error: null })).toBe("idle");
    expect(computeIndicatorState({ active: false, ready: false, error: "boom" })).toBe("idle");
    expect(computeIndicatorState({ active: false, ready: true, error: "boom" })).toBe("idle");
  });

  it("active + error yields error regardless of ready", () => {
    expect(computeIndicatorState({ active: true, ready: false, error: "boom" })).toBe("error");
    expect(computeIndicatorState({ active: true, ready: true, error: "boom" })).toBe("error");
  });

  it("active + ready + no error yields ready", () => {
    expect(computeIndicatorState({ active: true, ready: true, error: null })).toBe("ready");
  });

  it("active + neither ready nor errored yields loading", () => {
    expect(computeIndicatorState({ active: true, ready: false, error: null })).toBe("loading");
  });
});

describe("onIndicatorStateChange", () => {
  it("no error -> no error: no change", () => {
    expect(onIndicatorStateChange(null, null)).toBe(false);
  });

  it("no error -> new error: change", () => {
    expect(onIndicatorStateChange(null, "boom")).toBe(true);
  });

  it("same error repeated on a later poll: no change", () => {
    expect(onIndicatorStateChange("boom", "boom")).toBe(false);
  });

  it("different error replacing a prior one: change", () => {
    expect(onIndicatorStateChange("boom", "kaboom")).toBe(true);
  });

  it("recovery (error -> null): no change", () => {
    expect(onIndicatorStateChange("boom", null)).toBe(false);
  });
});

describe("isActivationKey", () => {
  it("Enter and Space are activation keys", () => {
    expect(isActivationKey("Enter")).toBe(true);
    expect(isActivationKey(" ")).toBe(true);
  });

  it("other keys are not activation keys", () => {
    expect(isActivationKey("Escape")).toBe(false);
    expect(isActivationKey("Tab")).toBe(false);
    expect(isActivationKey("a")).toBe(false);
  });
});
