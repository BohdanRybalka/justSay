/**
 * What the widget does about a recording it can no longer account for.
 *
 * ADR 049: a client-side abort stops the client waiting, it does not stop the
 * backend. Every mutating endpoint here answers *after* it has acted, so a
 * request that never answers leaves an outcome that is neither success nor
 * failure — this window knows only that it stopped listening, and the
 * microphone may be open with nothing in the app able to close it.
 *
 * The first version of this module could not fix that, because it was tracking
 * an obligation it could not name: `GET /audio/status` reported the one
 * process-wide recorder and said nothing about whose it was, so a compensating
 * stop could end the Settings microphone test or a dictation the user was in
 * the middle of speaking. Spec 119 gives every capture a client-minted id, so
 * the obligation is now per session and the question "may I end this" is
 * decided by the backend, inside the recorder's own lock, rather than guessed
 * here.
 *
 * `POST /audio/discard` is both the remedy and the probe. It writes no file —
 * a compensating stop used to leave audio of the room in the scratch directory
 * with nothing to remove it ([JS-122]) — and its answer is decisive in both
 * directions: 200 means the backend still held that session, so nothing
 * downstream of the start had run; 403 and 409 mean it did not.
 */

import { isDecisiveRefusal } from "../session";

/** What became of a session this window could not account for.
 *
 *  `proven` is the single outcome that establishes the backend was still
 *  holding it when the probe arrived, which is what a caller racing a request
 *  against the promise is asking about. `not-live` is every other way an entry
 *  leaves the map — the recorder answered that this session is not the live
 *  capture, or the request it belonged to answered after all — and it exists so
 *  that removing an entry always settles the promise `owe()` handed out.
 *  Deleting the key without settling retained the caller's reaction for the
 *  window's life, one dead closure per dictation. */
export type OwedOutcome = "proven" | "not-live";

/** One session this window cannot account for. `settleWith` settles the promise
 *  `owe()` handed back, and is called exactly once, by whichever removal took
 *  the entry out of the map. */
interface OwedSession {
  probeNotBefore: number;
  promise: Promise<OwedOutcome>;
  settleWith: (outcome: OwedOutcome) => void;
}

export type SettleOutcome = "settled" | "deferred" | "nothing-owed";

export interface AbandonedSessions {
  owe(sessionId: string, probeNotBefore: number): Promise<OwedOutcome>;
  forget(sessionId: string): void;
  settle(now: number): Promise<SettleOutcome>;
}

/**
 * Per-session accounting for requests this window abandoned.
 *
 * Keyed by session id rather than held as a boolean, which is what closes the
 * lost update the fourth review pass of spec 113 found: a settle used to read
 * the debt before its await and write it back after, so an obligation recorded
 * while a probe was in flight was erased by that probe's completion. A settle
 * here removes **the key it settled** and there is no whole-collection write
 * anywhere in this module, so an entry added during the await is simply a
 * different key.
 *
 * Single-flight by design. The connection poll fires every 5 s while a discard
 * carries a 15 s budget, so without it three probes would be in the air at
 * once against a backend already believed dead. Draining one session per poll
 * is deliberate and recorded as a deferred cut in the spec.
 */
export function createAbandonedSessions(deps: {
  discard: (sessionId: string) => Promise<unknown>;
}): AbandonedSessions {
  const owed = new Map<string, OwedSession>();
  let probeInFlight = false;

  function dueSession(now: number): [string, OwedSession] | null {
    let oldest: [string, OwedSession] | null = null;
    for (const entry of owed) {
      if (entry[1].probeNotBefore > now) continue;
      if (!oldest || entry[1].probeNotBefore < oldest[1].probeNotBefore) oldest = entry;
    }
    return oldest;
  }

  return {
    /** Record a session, and hand back a promise that resolves only once a
     *  probe has proved the backend still held it.
     *
     *  `probeNotBefore` is what keeps the probe from beating a healthy handler
     *  to the recorder's lock: `POST /pipeline/dictate` stops the recorder as
     *  its first act, so a discard sent immediately could find the session
     *  still there and destroy a dictation that was about to be transcribed.
     *  An abandoned start faces the same race — nothing serializes a queued
     *  `POST /audio/start` against the rest of the event loop — so every caller
     *  in this window sets the same wait.
     *
     *  Re-owing a session already recorded keeps the original entry and its
     *  promise, so a caller cannot end up holding a promise nothing will ever
     *  resolve. */
    owe(sessionId, probeNotBefore) {
      const existing = owed.get(sessionId);
      if (existing) return existing.promise;

      let settleWith!: (outcome: OwedOutcome) => void;
      const promise = new Promise<OwedOutcome>((resolve) => {
        settleWith = resolve;
      });
      owed.set(sessionId, { probeNotBefore, promise, settleWith });
      return promise;
    },

    /** The request answered, so there is nothing left to reconcile. The promise
     *  settles as `not-live`: the session is accounted for, and nothing here
     *  proved the backend was still holding it. */
    forget(sessionId) {
      const session = owed.get(sessionId);
      if (!session) return;
      owed.delete(sessionId);
      session.settleWith("not-live");
    },

    /** Send at most one discard, for the oldest session whose wait has passed.
     *
     *  An entry is removed only on an answer the *recorder* gave, from inside
     *  its own lock, about this session:
     *
     *  - `200` — it held the session and has now released it, so nothing
     *    downstream of the abandoned start ever ran.
     *  - `403` — it holds a different session, so this one is not live.
     *  - `409` — it holds nothing, so this one is not live.
     *
     *  Everything else keeps the entry, and the next poll probes again. A
     *  `TimedOutError` or a transport failure is more silence. A `401` is the
     *  auth middleware refusing the request *before* the route handler runs
     *  (`backend/app/core/auth_middleware.py`), and `src/api.ts` maps every
     *  `401` to `ApiAuthError` ahead of any endpoint-specific logic, so a
     *  stale launch token would otherwise discharge every owed session at once
     *  while the microphone stayed open — the [JS-121]/[JS-122] failure this
     *  module exists to close. A `422`, a `404` or a `500` raised before the
     *  handler reaches the lock are refusals of the *request*, not answers
     *  about the session, and are treated the same way. Retrying one of these
     *  costs a probe per poll; discharging one wrongly costs the obligation
     *  permanently, and only the second failure is unrecoverable. */
    async settle(now) {
      if (owed.size === 0) return "nothing-owed";
      if (probeInFlight) return "deferred";

      const due = dueSession(now);
      if (!due) return "deferred";
      const [sessionId, session] = due;

      probeInFlight = true;
      try {
        await deps.discard(sessionId);
      } catch (e) {
        if (!isDecisiveRefusal(e)) return "deferred";
        owed.delete(sessionId);
        session.settleWith("not-live");
        return "settled";
      } finally {
        probeInFlight = false;
      }

      owed.delete(sessionId);
      session.settleWith("proven");
      return "settled";
    },
  };
}
