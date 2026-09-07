// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiAuthError, ApiRequestError } from "../api";
import { TimedOutError } from "../timeout";
import { createAbandonedSessions } from "./abandoned-request";

const SESSION_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SESSION_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/** A discard whose answer each test releases by hand, so the window between a
 *  probe going out and its answer arriving is a place assertions can be made
 *  rather than a race. */
function controllableDiscard() {
  const releases: Array<{ sessionId: string; resolve: () => void; reject: (e: unknown) => void }> =
    [];
  const discard = vi.fn(
    (sessionId: string) =>
      new Promise<unknown>((resolve, reject) => {
        releases.push({ sessionId, resolve: () => resolve({}), reject });
      }),
  );
  return { discard, releases };
}

describe("createAbandonedSessions", () => {
  let discard: ReturnType<typeof controllableDiscard>["discard"];
  let releases: ReturnType<typeof controllableDiscard>["releases"];

  beforeEach(() => {
    ({ discard, releases } = controllableDiscard());
  });

  it("owes nothing until a session is recorded", async () => {
    const sessions = createAbandonedSessions({ discard });

    await expect(sessions.settle(0)).resolves.toBe("nothing-owed");
    expect(discard).not.toHaveBeenCalled();
  });

  it("holds a session back until its earliest probe time has passed", async () => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_A, 15_000);

    await expect(sessions.settle(14_999)).resolves.toBe("deferred");
    expect(discard).not.toHaveBeenCalled();

    void sessions.settle(15_000);
    expect(discard).toHaveBeenCalledWith(SESSION_A);
  });

  it("resolves the promise it handed out only when a probe answers 200", async () => {
    const sessions = createAbandonedSessions({ discard });
    const proven = vi.fn();
    void sessions.owe(SESSION_A, 0).then(proven);

    const settling = sessions.settle(0);
    releases[0].resolve();
    await expect(settling).resolves.toBe("settled");
    await Promise.resolve();

    expect(proven).toHaveBeenCalled();
  });

  it.each([
    ["a 403, because the recording is somebody else's", new ApiRequestError("not yours", 403)],
    ["a 409, because nothing is recording", new ApiRequestError("not recording", 409)],
  ])("drops a session on %s", async (_name, answer) => {
    const sessions = createAbandonedSessions({ discard });
    const proven = vi.fn();
    void sessions.owe(SESSION_A, 0).then(proven);

    const settling = sessions.settle(0);
    releases[0].reject(answer);
    await expect(settling).resolves.toBe("settled");
    await Promise.resolve();

    await expect(sessions.settle(0)).resolves.toBe("nothing-owed");
    expect(proven).not.toHaveBeenCalled();
  });

  it.each([
    ["a timeout, because that is more silence", new TimedOutError(15_000, "/audio/discard")],
    ["a transport failure, because nothing was established", new TypeError("Failed to fetch")],
    [
      "a 401, because the auth middleware answers before the recorder is reached",
      new ApiAuthError("no token", { kind: "ok" }),
    ],
    [
      "a 500, because a failure before the lock says nothing about the session",
      new ApiRequestError("boom", 500),
    ],
  ])("keeps a session on %s", async (_name, failure) => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_A, 0);

    const settling = sessions.settle(0);
    releases[0].reject(failure);
    await expect(settling).resolves.toBe("deferred");

    void sessions.settle(0);
    expect(discard).toHaveBeenCalledTimes(2);
  });

  it("keeps a session recorded during a probe for a different one", async () => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_A, 0);

    const settling = sessions.settle(0);
    expect(discard).toHaveBeenCalledWith(SESSION_A);

    void sessions.owe(SESSION_B, 0);
    releases[0].resolve();
    await expect(settling).resolves.toBe("settled");

    void sessions.settle(0);
    expect(discard).toHaveBeenNthCalledWith(2, SESSION_B);
  });

  it("drains two sessions owed before any probe, and proves each one", async () => {
    const sessions = createAbandonedSessions({ discard });
    const provenA = vi.fn();
    const provenB = vi.fn();
    void sessions.owe(SESSION_A, 0).then(provenA);
    void sessions.owe(SESSION_B, 1_000).then(provenB);

    const first = sessions.settle(1_000);
    expect(discard).toHaveBeenNthCalledWith(1, SESSION_A);
    releases[0].resolve();
    await expect(first).resolves.toBe("settled");

    const second = sessions.settle(1_000);
    expect(discard).toHaveBeenNthCalledWith(2, SESSION_B);
    releases[1].resolve();
    await expect(second).resolves.toBe("settled");
    await Promise.resolve();

    expect(provenA).toHaveBeenCalled();
    expect(provenB).toHaveBeenCalled();
    await expect(sessions.settle(1_000)).resolves.toBe("nothing-owed");
  });

  it("sends one probe at a time, however often the poll fires", async () => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_A, 0);
    void sessions.owe(SESSION_B, 0);

    void sessions.settle(0);
    await expect(sessions.settle(0)).resolves.toBe("deferred");

    expect(discard).toHaveBeenCalledTimes(1);
  });

  it("forgets a session whose request answered", async () => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_A, 0);

    sessions.forget(SESSION_A);

    await expect(sessions.settle(0)).resolves.toBe("nothing-owed");
    expect(discard).not.toHaveBeenCalled();
  });

  it("hands a re-owed session the promise it already gave out", async () => {
    const sessions = createAbandonedSessions({ discard });
    const first = vi.fn();
    const second = vi.fn();
    void sessions.owe(SESSION_A, 0).then(first);
    void sessions.owe(SESSION_A, 99_999).then(second);

    const settling = sessions.settle(0);
    releases[0].resolve();
    await settling;
    await Promise.resolve();

    expect(first).toHaveBeenCalled();
    expect(second).toHaveBeenCalled();
  });

  it("probes the session that has been due longest", async () => {
    const sessions = createAbandonedSessions({ discard });
    void sessions.owe(SESSION_B, 5_000);
    void sessions.owe(SESSION_A, 1_000);

    void sessions.settle(10_000);

    expect(discard).toHaveBeenCalledWith(SESSION_A);
  });
});
