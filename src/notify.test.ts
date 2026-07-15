import { describe, expect, it } from "vitest";
import { nextConnectionCheckState, onConnectivityChange } from "./notify";

describe("onConnectivityChange", () => {
  it("online -> online: stays online, no notify", () => {
    expect(onConnectivityChange(false, false)).toEqual({ offline: false, shouldNotify: false });
  });

  it("online -> offline: edge triggers notify", () => {
    expect(onConnectivityChange(false, true)).toEqual({ offline: true, shouldNotify: true });
  });

  it("offline -> offline: repeated poll, no re-notify", () => {
    expect(onConnectivityChange(true, true)).toEqual({ offline: true, shouldNotify: false });
  });

  it("offline -> online: recovery resets the edge, no notify", () => {
    expect(onConnectivityChange(true, false)).toEqual({ offline: false, shouldNotify: false });
  });
});

describe("nextConnectionCheckState", () => {
  it("first-ever call succeeding: no notify, firstCheckDone flips true", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: false }, true)).toEqual({
      offline: false,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("first-ever call failing: no notify despite the offline edge, firstCheckDone flips true", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: false }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("a later failure after a prior success: notifies", () => {
    expect(nextConnectionCheckState({ offline: false, firstCheckDone: true }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: true,
    });
  });

  it("a continued outage: no repeat notify", () => {
    expect(nextConnectionCheckState({ offline: true, firstCheckDone: true }, false)).toEqual({
      offline: true,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });

  it("recovery: offline flips back to false, no notify", () => {
    expect(nextConnectionCheckState({ offline: true, firstCheckDone: true }, true)).toEqual({
      offline: false,
      firstCheckDone: true,
      shouldNotify: false,
    });
  });
});
