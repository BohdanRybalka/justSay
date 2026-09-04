import { ApiAuthError } from "../api";
import { TimedOutError } from "../timeout";

export interface DictationErrorLabel {
  /** Compact widget text — the pill is ~240 px wide, so keep it short. */
  label: string;
  toast: string;
}

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
 * Two failures get their own wording, because each is a claim the start's
 * generic text would get wrong. An abandoned request: the handler opens the
 * device and then answers, so a start that ran out of its budget may have left
 * the microphone open and nothing this window can read says whether it did. And
 * a 401: `request()` builds the token header for `/audio/start` exactly as it
 * does for `/pipeline/dictate`, so the same authentication fault reaches here —
 * and "try again" is advice that fails identically every time it is taken.
 * Every other failure keeps the start's own generic text.
 */
export function startErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return {
      label: "Auth failed",
      toast: "JustSay could not authenticate to its own backend — restart the app.",
    };
  }

  if (error instanceof TimedOutError) {
    return {
      label: "No answer",
      toast: "The backend never answered — the microphone may still be open.",
    };
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
 *
 * The `TimedOutError` branch runs before it for the same reason and one of its
 * own: "Failed" would be a claim the widget cannot make. Two separate things
 * are unknown after an abandoned `POST /pipeline/dictate`, and the toast names
 * both. A cancellation inside the pipeline can land either side of the
 * clipboard write, and nothing the frontend can read says which. And the stop
 * that ends the capture is the handler's own first act
 * (`backend/app/pipeline/router.py:45-48`), so a backend that never ran the
 * handler never stopped the recorder either, and nothing in this window can
 * close it — which is what the toast says rather than glossing.
 */
export function dictationErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return {
      label: "Auth failed",
      toast: "JustSay could not authenticate to its own backend — restart the app.",
    };
  }

  if (error instanceof TimedOutError) {
    return {
      label: "No answer",
      toast:
        "The backend never answered — your text may not have been copied, and the microphone may still be open.",
    };
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
