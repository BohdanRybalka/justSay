import { describe, expect, it, vi } from "vitest";

vi.mock("../settings", () => ({ loadSettings: vi.fn() }));

import { isStaleStatusResponse } from "./models";

describe("isStaleStatusResponse", () => {
  it("a token superseded by a newer issued token is stale", () => {
    const latestIssuedToken = 2;
    expect(isStaleStatusResponse(1, latestIssuedToken)).toBe(true);
  });

  it("the most recently issued token is never stale", () => {
    const latestIssuedToken = 2;
    expect(isStaleStatusResponse(2, latestIssuedToken)).toBe(false);
  });

  it("a single in-flight request with no newer one issued is never stale", () => {
    const latestIssuedToken = 1;
    expect(isStaleStatusResponse(1, latestIssuedToken)).toBe(false);
  });
});
