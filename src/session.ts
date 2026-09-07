/**
 * The name a window gives a recording before it asks for one.
 *
 * `POST /audio/start` used to answer with nothing that identified what it had
 * started, so "is this recording mine" had no answer and every recovery built
 * on top of it was a guess about a process-wide recorder three surfaces can
 * drive. Minting the id here — before the request goes out — is what turns
 * that guess into a comparison
 * (docs/adr/050-the-client-names-the-recording-before-it-asks-for-one.md).
 */

import { ApiRequestError } from "./api";

const SESSION_ID_BYTES = 16;

/** 32 lowercase hex characters from `crypto.getRandomValues`.
 *
 *  Not `crypto.randomUUID`, which the platform gates on a secure context: the
 *  app is served from a custom scheme on macOS and the dev-open path is plain
 *  `http://localhost`, so `randomUUID` is absent in exactly the places this has
 *  to work. `getRandomValues` carries no such gate.
 *
 *  Randomness rather than a counter because the ids must not collide across
 *  windows or across a reload, and neither of those shares a counter. 128 bits
 *  is far more than the handful of live sessions this app ever has; the width
 *  is set by `SESSION_ID_PATTERN`, which the backend validates against. */
export function newSessionId(): string {
  const bytes = new Uint8Array(SESSION_ID_BYTES);
  crypto.getRandomValues(bytes);
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** The two refusals a session-guarded endpoint can only produce from inside the
 *  recorder's lock, having compared the caller's id against the live one:
 *  `403` (the recorder is held by a different session) and `409` (nothing is
 *  being recorded). Both are decisive in the same direction — the named session
 *  is not the live capture — which is what separates them from every other
 *  status a caller can see. Those are produced before the route handler runs,
 *  or by something other than that comparison, and so say nothing about who
 *  holds the recorder. */
const DECISIVE_REFUSALS = new Set([403, 409]);

/** Whether a rejection is the recorder's own answer that the named session is
 *  not live, as opposed to more silence. */
export function isDecisiveRefusal(error: unknown): boolean {
  return error instanceof ApiRequestError && DECISIVE_REFUSALS.has(error.status);
}
