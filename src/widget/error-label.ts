import { ApiAuthError } from "../api";

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
 * that function's other two branches actively misdescribe one. A `409 Already
 * recording` is not "Dictation failed — try again", and the substring heuristic
 * reads the word "missing" out of a microphone fault and sends the user to the
 * cloud-key screen — the exact spec 042 defect, reintroduced at a different
 * endpoint by routing the start through the dictation labels.
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

/**
 * Decides what the widget shows after a failed dictation.
 *
 * The `ApiAuthError` branch runs BEFORE the substring heuristic on purpose: the
 * backend's 401 body is `"Missing or invalid API token"`, which contains
 * "missing" and therefore used to render "Add key in Settings" — sending the
 * user to add a cloud API key when the actual failure was the app not
 * authenticating to its own local backend (spec 042).
 */
export function dictationErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return AUTH_FAILED;
  }

  const msg = (error instanceof Error ? error.message : String(error)).toLowerCase();
  if (msg.includes("missing")) {
    return {
      label: "Add key in Settings",
      toast: "No API key set — add one in Settings.",
    };
  }

  return { label: "Failed", toast: "Dictation failed — try again." };
}
