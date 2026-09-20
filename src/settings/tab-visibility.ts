/**
 * The one answer to "the window's visibility moved — what does the tab do?".
 *
 * The shell states both edges rather than letting the page infer them (ADR
 * 089), and either can arrive when nothing actually changed: the tray's
 * Settings item shows a window that is already visible, so a second `shown` in
 * a row is ordinary traffic. Reducing the pair to an action in one pure
 * function gives that idempotence a single owner, which is why a tab's
 * `releaseResources` and `resumeResources` carry no guard of their own.
 *
 * It sits in its own file because importing `settings.ts` boots the window.
 */
export type TabVisibilityAction = "release" | "resume" | "ignore";

export function nextTabAction(
  event: "hidden" | "shown",
  wasHidden: boolean,
): TabVisibilityAction {
  if (event === "hidden") return wasHidden ? "ignore" : "release";
  return wasHidden ? "resume" : "ignore";
}
