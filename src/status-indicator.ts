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
// (plus, when interactive, the role/tabindex/aria-label a keyboard-operable
// badge needs) only. All visual behavior (spinner animation, checkmark
// glyph, error glyph, hidden-when-idle) lives in CSS, keyed off the modifier
// class.
export function renderIndicator(
  el: HTMLElement,
  state: IndicatorState,
  opts?: { title?: string; interactive?: boolean; ariaLabel?: string },
): void {
  el.className = `status-indicator-badge status-indicator-badge--${state}`;
  el.title = opts?.title ?? "";
  if (opts?.interactive) {
    el.setAttribute("role", "button");
    el.setAttribute("tabindex", "0");
    el.setAttribute("aria-label", opts?.ariaLabel ?? opts?.title ?? "Retry");
  } else {
    el.removeAttribute("role");
    el.removeAttribute("tabindex");
    el.removeAttribute("aria-label");
  }
}

// Pure — extracted so it's unit-testable without a DOM/jsdom dependency
// (ADR 003's pure-function-only Vitest scope).
export function isActivationKey(key: string): boolean {
  return key === "Enter" || key === " ";
}

// Wires click + keydown(Enter/Space) to one activation handler, mirroring
// the existing dropzone pattern (src/settings/tabs/transcribe.ts). Re-reads
// role="button" on every event (not once at bind time) so whatever the most
// recent renderIndicator({interactive}) call decided is the single source
// of truth for "should this actually fire" — this is what replaces
// models.ts's old classList.contains("status-indicator-badge--error")
// string check with one guard that can't drift out of sync with the
// rendered state.
export function bindIndicatorActivation(el: HTMLElement, onActivate: () => void): void {
  const isInteractive = () => el.getAttribute("role") === "button";
  el.addEventListener("click", (e) => {
    if (!isInteractive()) return;
    e.stopPropagation();
    onActivate();
  });
  el.addEventListener("keydown", (e) => {
    if (!isInteractive() || !isActivationKey(e.key)) return;
    e.preventDefault();
    e.stopPropagation();
    onActivate();
  });
}
