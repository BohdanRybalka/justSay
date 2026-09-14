import { ApiAuthError, ApiRequestError, CONFIGURATION_ERROR_CODE } from "../api";

export interface DictationErrorLabel {
  /** Compact widget text — the pill is ~240 px wide, so keep it short. */
  label: string;
  toast: string;
}

/** The same fault reaches both functions below through the same header:
 *  `request()` builds the token header for `/audio/start` exactly as it does
 *  for `/pipeline/dictate`, so a 401 is one failure with one remedy, and two
 *  copies of the sentence are two places for it to drift. */
const AUTH_FAILED: DictationErrorLabel = {
  label: "Auth failed",
  toast: "JustSay could not authenticate to its own backend — restart the app.",
};

/**
 * Decides what the widget shows after a failed `POST /audio/start`.
 *
 * Separate from `dictationErrorLabel` because a start is not a dictation and
 * that function's other branches actively misdescribe one. A `409 Already
 * recording` is not "Dictation failed — try again", and a start has no
 * remedial key to add, so the labels that endpoint needs are its own.
 *
 * The 401 gets its own wording because "try again" is advice that fails
 * identically every time it is taken. Every other failure keeps the start's own
 * generic text.
 */
export function startErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return AUTH_FAILED;
  }

  return { label: "Start failed", toast: "Couldn't start recording — try again." };
}

/** Something else is holding the microphone, and this window may not end it.
 *
 *  A new outcome as of spec 119: the recorder answers 403 to a session that
 *  does not own it, which happens when the Settings microphone test or another
 *  window took the device between this dictation's start and its stop. Naming
 *  it is the whole point — "Dictation failed — try again" invites a retry that
 *  fails identically until the other surface lets go. */
const NOT_YOURS: DictationErrorLabel = {
  label: "Recording is busy",
  toast: "Another window is using the microphone — stop it there and try again.",
};

/** The dictation was never processed, and that is now a fact rather than a
 *  guess.
 *
 *  `POST /pipeline/dictate` stops the recorder as its first act, so a session
 *  the backend was still holding when `POST /audio/discard` answered 200 is a
 *  dictation whose handler had not run. The discard also ended the capture, so
 *  the handler now cannot run: nothing was transcribed, nothing was copied and
 *  the microphone is closed. Before spec 119 this case sat in `processing` for
 *  the 600 s dictation budget with the recorder still filling its buffer. */
export const DICTATION_NEVER_PROCESSED: DictationErrorLabel = {
  label: "No answer",
  toast: "The backend never answered — nothing was transcribed and the microphone is closed.",
};

/**
 * Decides what the widget shows after a failed dictation.
 *
 * Every branch reads a status or a `code` and none reads the message, which is
 * what keeps two defects closed structurally rather than by ordering. The
 * backend's 401 body is `"Missing or invalid API token"` and the 403's names a
 * missing owning session; both used to match a substring test for the word
 * "missing" and render "Add key in Settings", sending the user to add a cloud
 * API key for a failure that was the app not authenticating to its own local
 * backend (spec 042) or another window holding the microphone (spec 119).
 * Neither body is consulted now, so neither can select a label again.
 */
export function dictationErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return AUTH_FAILED;
  }

  if (error instanceof ApiRequestError && error.status === 403) {
    return NOT_YOURS;
  }

  if (error instanceof ApiRequestError && error.code === CONFIGURATION_ERROR_CODE) {
    return {
      label: "Add key in Settings",
      toast: "No API key set — add one in Settings.",
    };
  }

  return { label: "Failed", toast: "Dictation failed — try again." };
}
