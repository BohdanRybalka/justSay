import { describe, expect, it, vi } from "vitest";

// models.ts transitively imports ../settings, whose top-level code touches
// `document` (e.g. `document.getElementById("tab-content")`) -- this project
// has no jsdom test environment (tracked in docs/TODO.md), so importing
// models.ts unmocked would throw "document is not defined" before any test
// body runs. Stubbed here purely so the module can be imported for its one
// pure export under test; none of these mocked bindings are exercised.
vi.mock("../settings", () => ({ loadSettings: vi.fn() }));

import { isStaleStatusResponse } from "./models";

describe("isStaleStatusResponse", () => {
  it("a token superseded by a newer issued token is stale", () => {
    // Token 1 issued, then token 2 issued before token 1's request resolves.
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
