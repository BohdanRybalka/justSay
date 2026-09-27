import { describe, expect, it } from "vitest";
import { backendStateOf } from "./sidebar";

describe("backendStateOf", () => {
  it.each([
    [{ reachable: true, refusedRequest: false, starting: true }, "ready"],
    [{ reachable: true, refusedRequest: false, starting: false }, "ready"],
    [{ reachable: true, refusedRequest: true, starting: false }, "unauthorized"],
    [{ reachable: false, refusedRequest: false, starting: true }, "starting"],
    [{ reachable: false, refusedRequest: false, starting: false }, "offline"],
    [{ reachable: false, refusedRequest: true, starting: false }, "offline"],
  ])("reads %j as %s", (backend, state) => {
    expect(backendStateOf(backend)).toBe(state);
  });
});
