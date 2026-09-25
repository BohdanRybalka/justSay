import type { DictateResponse } from "../api";
import type { PillView } from "./pill";

/**
 * What the pill shows once a dictation has answered and its text has gone to
 * the clipboard or failed to, or `null` for empty text with no discard reason.
 * Silence is checked first: a silence-guard discard also has empty text and
 * must still read "No speech". Words are counted by the backend's own
 * `word_count` rule, whitespace-separated runs.
 */
export function dictationResultView(
  result: Pick<DictateResponse, "text" | "discarded_reason">,
  copied: boolean,
): PillView | null {
  if (result.discarded_reason === "silence") return { kind: "noSpeech" };
  const text = result.text.trim();
  if (!text) return null;
  if (!copied) return { kind: "alert", label: "Copy failed" };
  return { kind: "done", words: text.split(/\s+/).length };
}
