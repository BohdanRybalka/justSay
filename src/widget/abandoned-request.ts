/**
 * What the widget does about a state-mutating request it abandoned.
 *
 * ADR 049: a client-side abort stops the client waiting, it does not stop the
 * backend. Every endpoint here answers *after* it has acted, so a budget that
 * expires leaves an outcome that is neither success nor failure — the widget
 * knows only that it stopped listening. Collapsing that into "failed" is what
 * put the widget into `error` with the microphone still open, which is the
 * original defect in a new shape.
 *
 * Two different questions come out of that, and they are answered in different
 * places. *What may the widget claim on screen* is answered once, here, by
 * reading the status endpoint that already exists and branching on three
 * outcomes rather than two. *What may the widget leave running* cannot be
 * answered by that read at all: the read fails for the same reason the budget
 * expired, and a decision that cannot be made now must still be made later. So
 * the second question becomes an obligation the widget records and discharges
 * on the connection poll it already runs.
 *
 * What neither can establish is ownership. `GET /audio/status` reports the one
 * process-wide recorder and says nothing about who started it, and there is no
 * session id on `/audio/start` to echo back. So a capture the widget adopts may
 * be somebody else's — Settings' microphone test drives the same recorder — and
 * a stop the widget owes can end that test rather than its own abandoned start.
 * Both are named risks rather than proofs: adopting is right about *whether* a
 * recording exists and only presumed about *whose* it is, and stopping is the
 * direction in which being wrong is survivable, because Settings swallows the
 * refusal while a microphone nothing can close records the room. [JS-119]
 * carries the session id that would settle it.
 */

import { ApiAuthError, ApiRequestError } from "../api";

export interface RecordingSnapshot {
  is_recording: boolean;
  duration_seconds: number;
}

/** What the backend turned out to be doing, with `unknown` for the case the
 *  status read could not answer either. `unknown` is a third value rather than
 *  a synonym for `idle` because the meeting indicator resolves it toward
 *  showing while the dictation widget adopts only a confirmed recording — the
 *  widget compares `kind === "recording"` inline, so anything short of a
 *  positive answer leaves the start failed and the stop owed. */
export type RecordingTruth =
  | { kind: "recording"; elapsedSeconds: number }
  | { kind: "idle" }
  | { kind: "unknown" };

export async function readRecordingTruth(
  read: () => Promise<RecordingSnapshot>,
): Promise<RecordingTruth> {
  try {
    const snapshot = await read();
    return snapshot.is_recording
      ? { kind: "recording", elapsedSeconds: snapshot.duration_seconds }
      : { kind: "idle" };
  } catch {
    return { kind: "unknown" };
  }
}

/** A meeting start that timed out. `unknown` resolves toward showing rather
 *  than hiding because a false negative breaks ADR 040 obligation 2 with no
 *  recovery until the window reloads, while a false positive is provisional:
 *  the widget marks the indicator unconfirmed and the connection poll withdraws
 *  it on the first status read that reports no recording. */
export function shouldShowIndicatorAfterAbandonedMeetingStart(truth: RecordingTruth): boolean {
  return truth.kind !== "idle";
}

/** The `POST /audio/stop` an abandoned dictation left behind — a start that
 *  could not be adopted, or a `POST /pipeline/dictate` whose own handler never
 *  ran the stop it opens with — held until the backend is answering again. */
export interface AbandonedStartCleanup {
  owe(): void;
  settle(backendAnswered: boolean): Promise<"nothing-owed" | "deferred" | "settled">;
}

/** The stop is owed by any abandoned start the widget did not adopt, `idle`
 *  answers included: the handler is cancelled either before the device open
 *  runs or after `_recording` is already true, and the status read is a
 *  separate request whose ordering against that resumption is not guaranteed.
 *  A needless stop costs one 409, which is itself an answer and clears the debt.
 *
 *  The debt survives everything except an answer. An unreachable backend, a
 *  busy widget and a stop still in flight all defer it — the poll fires every
 *  5 s while a stop carries a 15 s budget, so without the single-flight flag
 *  three could be in the air at once. A resolved stop clears it, and so does an
 *  `ApiRequestError` or an `ApiAuthError`, because both mean the backend saw
 *  the request. Any other rejection is another silence and keeps it.
 *
 *  A stop that lands writes a WAV. `POST /audio/stop`
 *  (`backend/app/audio/router.py:108-117`) harvests the capture and returns its
 *  filename, and the only path that deletes such a file is `/pipeline/dictate`
 *  (`backend/app/pipeline/router.py:71-73`), which this one is not. So every
 *  discharged obligation leaves one untranscribed recording in the temp
 *  directory that nothing announces and nothing removes until the user runs
 *  Settings' cleanup. It is an accepted cost rather than an oversight — no
 *  per-file delete endpoint exists and this spec is frontend-only — recorded in
 *  ADR 049 and filed as [JS-122]. The alternative is the open microphone this
 *  obligation exists to close. */
export function createAbandonedStartCleanup(deps: {
  stopRecording: () => Promise<unknown>;
  isBusy: () => boolean;
}): AbandonedStartCleanup {
  let owed = false;
  let stopInFlight = false;

  return {
    owe() {
      owed = true;
    },
    async settle(backendAnswered) {
      if (!owed) return "nothing-owed";
      if (!backendAnswered || stopInFlight || deps.isBusy()) return "deferred";

      stopInFlight = true;
      try {
        await deps.stopRecording();
      } catch (e) {
        if (!(e instanceof ApiRequestError || e instanceof ApiAuthError)) {
          return "deferred";
        }
      } finally {
        stopInFlight = false;
      }
      owed = false;
      return "settled";
    },
  };
}
