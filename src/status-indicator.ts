/**
 * Universal readiness-indicator component (ADR 009).
 *
 * Generic "is background feature X currently loading/ready/failed" state
 * machine, deliberately decoupled from any single feature — the Local STT
 * toggle (spec 015) is its first consumer, a future LLM Local-mode panel
 * its second. See docs/adr/009-universal-status-indicator.md for the full
 * design rationale.
 */

export type IndicatorState = "idle" | "loading" | "ready" | "error";

// Pure. `active: false` always wins regardless of `ready`/`error` — this is
// what makes "switch back to Cloud => no leftover Local-loading badge" a
// structural guarantee instead of a timer or a manually-cleared flag.
export function computeIndicatorState(input: {
  active: boolean;
  ready: boolean;
  error: string | null;
}): IndicatorState {
  if (!input.active) return "idle";
  if (input.error) return "error";
  if (input.ready) return "ready";
  return "loading";
}

// Pure, edge-triggered exactly like notify.ts's onConnectivityChange: true
// only when a *new, different* error string appears, false on a repeated
// poll of the same error and on recovery (transition back to null).
export function onIndicatorStateChange(
  prevError: string | null,
  nextError: string | null,
): boolean {
  return nextError !== null && nextError !== prevError;
}

// The only DOM-touching piece, and intentionally tiny: sets className/title
// only. All visual behavior (spinner animation, checkmark glyph, error
// glyph, hidden-when-idle) lives in CSS, keyed off the modifier class.
export function renderIndicator(
  el: HTMLElement,
  state: IndicatorState,
  opts?: { title?: string },
): void {
  el.className = `status-indicator-badge status-indicator-badge--${state}`;
  el.title = opts?.title ?? "";
}
