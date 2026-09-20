/**
 * The one answer to "a visibility edge arrived — what does the tab do?".
 *
 * The edges are the shell's dismissal and show announcements (ADR 089), either
 * of which can repeat with nothing having changed, so the repeat is what this
 * reduces away.
 */
export type TabVisibilityAction = "release" | "resume" | "ignore";

export function nextTabAction(
  event: "hidden" | "shown",
  wasHidden: boolean,
): TabVisibilityAction {
  if (event === "hidden") return wasHidden ? "ignore" : "release";
  return wasHidden ? "resume" : "ignore";
}
