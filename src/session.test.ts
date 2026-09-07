// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { SESSION_ID_PATTERN } from "./contracts";
import { newSessionId } from "./session";

describe("newSessionId", () => {
  it("mints 100 ids that all match the pattern the backend validates against", () => {
    const minted = Array.from({ length: 100 }, () => newSessionId());

    const wrong = minted.filter((id) => !SESSION_ID_PATTERN.test(id));
    expect(wrong).toEqual([]);
  });

  it("mints 100 distinct ids, because a collision is one window ending another's recording", () => {
    const minted = Array.from({ length: 100 }, () => newSessionId());

    expect(new Set(minted).size).toBe(100);
  });

  it("pads a zero byte rather than dropping a character", () => {
    const zeroed = new Uint8Array(16);
    const original = crypto.getRandomValues.bind(crypto);
    crypto.getRandomValues = ((array: Uint8Array) => {
      array.set(zeroed);
      return array;
    }) as typeof crypto.getRandomValues;

    try {
      expect(newSessionId()).toBe("0".repeat(32));
    } finally {
      crypto.getRandomValues = original;
    }
  });
});
