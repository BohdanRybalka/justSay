import { describe, expect, it } from "vitest";
import { onConnectivityChange } from "./notify";

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
