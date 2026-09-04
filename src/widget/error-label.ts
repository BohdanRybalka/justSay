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
 * own: "Failed" would be a claim the widget cannot make. `POST /pipeline/dictate`
 * stops the recorder before it transcribes, so nothing is left open, but a
 * cancellation inside the pipeline can land either side of the clipboard write
 * and nothing the frontend can read says which (ADR 049). The toast therefore
 * says the text may not have been copied rather than that the dictation failed.
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
      toast: "The backend never answered — your text may not have been copied.",
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
