import { ApiAuthError, ApiRequestError, CONFIGURATION_ERROR_CODE } from "../api";
import { TimedOutError } from "../timeout";

export interface DictationErrorLabel {
  /** One of the pill's short labels; the toast carries the full sentence. */
  label: string;
  toast: string;
}

const NO_CONNECTION = "No connection";

const DIDNT_WORK = "Didn't work";

/** The same fault reaches both functions below through the same header:
 *  `request()` builds the token header for `/audio/start` exactly as it does
 *  for `/pipeline/dictate`, so a 401 is one failure with one remedy, and two
 *  copies of the sentence are two places for it to drift. */
const AUTH_FAILED: DictationErrorLabel = {
  label: NO_CONNECTION,
  toast: "JustSay could not authenticate to its own backend — restart the app.",
};

/** Something else is holding the microphone, and this window may not end it.
 *
 *  The recorder answers 403 to a dictation whose session it no longer holds,
 *  and 409 to a start while it is already recording — both mean the Settings
 *  microphone test or another window has the device. Naming it is the whole
 *  point: "try again" invites a retry that fails until the other one lets go. */
const MIC_BUSY: DictationErrorLabel = {
  label: "Mic is busy",
  toast: "Another window is using the microphone — stop it there and try again.",
};

/** The dictation was never processed, and that is now a fact rather than a
 *  guess: the backend still held the session when `POST /audio/discard`
 *  answered 200, so the dictate handler had not run and now cannot — nothing
 *  was transcribed, nothing was copied and the microphone is closed. */
export const DICTATION_NEVER_PROCESSED: DictationErrorLabel = {
  label: NO_CONNECTION,
  toast: "The backend never answered — nothing was transcribed and the microphone is closed.",
};

/** No answer at all: `fetch` rejects a refused or dropped connection with a
 *  `TypeError`, and a budget that ran out is a `TimedOutError`. */
function isUnanswered(error: unknown): boolean {
  return error instanceof TypeError || error instanceof TimedOutError;
}

/**
 * Decides what the widget shows after a failed `POST /audio/start`.
 *
 * Separate from `dictationErrorLabel` because a start is not a dictation: it
 * has no key to add, and its toast says the recording never began.
 */
export function startErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) return AUTH_FAILED;
  if (error instanceof ApiRequestError && error.status === 409) return MIC_BUSY;

  const toast = "Couldn't start recording — try again.";
  return { label: isUnanswered(error) ? NO_CONNECTION : DIDNT_WORK, toast };
}

/**
 * Decides what the widget shows after a failed dictation.
 *
 * Every branch reads a status, a `code` or the error's type and none reads the
 * message: the backend's 401 and 403 bodies both contain "missing", and a
 * substring test once sent the user to add a cloud API key for either (ADR 063).
 */
export function dictationErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) return AUTH_FAILED;
  if (error instanceof ApiRequestError && error.status === 403) return MIC_BUSY;

  if (error instanceof ApiRequestError && error.code === CONFIGURATION_ERROR_CODE) {
    return { label: "Add an API key", toast: "No API key set — add one in Settings." };
  }

  const toast = "Dictation failed — try again.";
  return { label: isUnanswered(error) ? NO_CONNECTION : DIDNT_WORK, toast };
}
