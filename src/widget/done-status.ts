import type { DictateResponse } from "../api";

export interface DoneStatus {
  label: "Copied" | "Copy failed" | "No speech";
  elapsedSeconds: number;
}

/**
 * Decides what the widget's compact "done" status should show after a dictation
 * completes, or `null` when there's nothing worth showing (a whitespace-only/empty
 * transcript with no discard reason — treated as "nothing happened," the widget
 * reverts straight to idle with no status and no route badge).
 *
 * The `discarded_reason === "silence"` branch runs BEFORE the empty-text check:
 * a silence-guard discard also has empty text, but it must render "No speech"
 * (a normal, non-error "done" outcome — the guard's whole point is to make an
 * otherwise-invisible discard visible), not fall through to `null`.
 */
export function computeDoneStatus(
  result: Pick<DictateResponse, "text" | "duration_ms" | "copied_to_clipboard" | "discarded_reason">,
): DoneStatus | null {
  if (result.discarded_reason === "silence") {
    return { label: "No speech", elapsedSeconds: result.duration_ms / 1000 };
  }
  if (!result.text.trim()) return null;
  return {
    label: result.copied_to_clipboard ? "Copied" : "Copy failed",
    elapsedSeconds: result.duration_ms / 1000,
  };
}
