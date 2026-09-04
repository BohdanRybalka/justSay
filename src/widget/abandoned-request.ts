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
 * The answer is to ask the status endpoint that already exists, once, and
 * branch on three outcomes rather than two. It is one read and not a poll
 * because one read answers the only question there is; and it stays here, at
 * the timeout site, rather than in the widget's connection poll, because only
 * here does the widget know the recording it is looking at is the one it asked
 * for — the Settings window's microphone test drives the same shared recorder.
 */

export interface RecordingSnapshot {
  is_recording: boolean;
  duration_seconds: number;
}

/** What the backend turned out to be doing, with `unknown` for the case the
 *  status read could not answer either. `unknown` is a third value rather than
 *  a synonym for `idle` because the two decisions below resolve it in opposite
 *  directions, and each does so for its own reason. */
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

/** A dictation start that timed out: adopt the recording if the backend really
 *  holds one, because the widget already told the user it was recording and it
 *  was — the audio captured while the budget ran out is real. `unknown`
 *  resolves to `error`: without a confirmed recording there is nothing to adopt
 *  and claiming one would leave a stopwatch running against no capture. */
export function stateAfterAbandonedStart(truth: RecordingTruth): "recording" | "error" {
  return truth.kind === "recording" ? "recording" : "error";
}

/** A meeting start that timed out. `unknown` resolves toward showing rather
 *  than hiding: a false positive is cleared by one press through a branch that
 *  already exists — the next press takes the stop branch, gets a 409 or 410, and
 *  the indicator comes down — while a false negative breaks ADR 040 obligation
 *  2 with no recovery until the window reloads. */
export function indicatorAfterAbandonedMeetingStart(truth: RecordingTruth): "show" | "hide" {
  return truth.kind === "idle" ? "hide" : "show";
}
