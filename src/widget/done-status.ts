import type { DictateResponse } from "../api";

export interface DoneStatus {
  label: "Copied" | "Copy failed";
  elapsedSeconds: number;
}

/**
 * Decides what the widget's compact "done" status should show after a dictation
 * completes, or `null` when there's nothing worth showing (a whitespace-only/empty
 * transcript — treated as "nothing happened," the widget reverts straight to idle
 * with no status and no route badge).
 */
export function computeDoneStatus(
  result: Pick<DictateResponse, "text" | "duration_ms" | "copied_to_clipboard">,
): DoneStatus | null {
  if (!result.text.trim()) return null;
  return {
    label: result.copied_to_clipboard ? "Copied" : "Copy failed",
    elapsedSeconds: result.duration_ms / 1000,
  };
}
