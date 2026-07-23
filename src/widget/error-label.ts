import { ApiAuthError } from "../api";

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
 */
export function dictationErrorLabel(error: unknown): DictationErrorLabel {
  if (error instanceof ApiAuthError) {
    return {
      label: "Auth failed",
      toast: "JustSay could not authenticate to its own backend — restart the app.",
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
