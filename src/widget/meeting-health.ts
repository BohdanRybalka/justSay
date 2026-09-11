/**
 * What the widget must do with a meeting status it polled while it believes a
 * call is being recorded.
 *
 * It lives outside widget.ts for the reason meeting-toggle.ts does: the two
 * answers are opposite and neither is obvious. A backend that says it is no
 * longer recording ends the indicator, because the marker ADR 040 obligation 2
 * requires must describe something that is actually happening. A backend that
 * is still recording but reports a `capture_incident` keeps the marker up and
 * degrades it, because the call is still being captured and the user needs to
 * know which half of it is missing while there is still time to act.
 *
 * A poll that failed never reaches here at all. An unreachable backend is not
 * evidence that the capture ended — the same asymmetry meeting-toggle.ts
 * records for a failed stop — and ending the marker on it would hide an
 * indicator while a recording may still be running.
 *
 * The sentences are written here and the backend token is their key, which is
 * the split meeting-toggle.ts already uses: the backend names what happened,
 * and what the user reads is a product decision. One clause per incident, and
 * two sentences built from it — a call still running and a call already
 * stopped need the same fact in different tenses, and writing the fact twice
 * is how the two would come to disagree.
 */

import type { MeetingStatus } from "../api";
import type { CaptureIncident } from "../contracts";

export type MeetingHealthAction =
  | { kind: "keep" }
  | { kind: "degrade"; incident: string; message: string }
  | { kind: "end"; message: string };

/** Keyed by the token type rather than by `string`, so adding an incident to
 *  `CAPTURE_INCIDENTS` without a sentence for it fails `tsc` instead of
 *  reaching a user as the unknown-incident wording. */
const INCIDENT_CLAUSES: Record<CaptureIncident, string> = {
  microphone_stalled: "your microphone stopped sending audio",
  system_audio_ended: "the other side's audio stopped being captured",
  storage_failed: "the recording could not be written to disk and stopped growing",
  storage_low: "the disk ran low on space and the recording stopped growing",
  storage_backlog: "the disk could not keep up and some seconds were dropped",
};

const UNKNOWN_INCIDENT_CLAUSE = "something went wrong with the capture";

const CAPTURE_ENDED_MESSAGE =
  "This call is no longer being recorded — the recording ended without being stopped here.";

/** The bare fact behind an incident token, in the tense a caller can build on. */
export function describeMeetingIncident(incident: string): string {
  return INCIDENT_CLAUSES[incident as CaptureIncident] ?? UNKNOWN_INCIDENT_CLAUSE;
}

export function decideMeetingHealth(
  status: MeetingStatus,
  locallyActive: boolean,
): MeetingHealthAction {
  if (!locallyActive) return { kind: "keep" };
  if (!status.is_recording) return { kind: "end", message: CAPTURE_ENDED_MESSAGE };

  const incident = status.capture_incident;
  if (!incident) return { kind: "keep" };
  return {
    kind: "degrade",
    incident,
    message: `This call is still being recorded, but ${describeMeetingIncident(incident)}.`,
  };
}
