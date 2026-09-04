import { ApiAuthError } from "../api";
import { TimedOutError } from "../timeout";

export interface DictationErrorLabel {
  /** Compact widget text — the pill is ~240 px wide, so keep it short. */
  label: string;
  toast: string;
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
 * handler never stopped the recorder either — the microphone may still be open,
 * which is why `stopAndProcess` also records the stop it owes (ADR 049).
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
