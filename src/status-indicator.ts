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

export function onIndicatorStateChange(
  prevError: string | null,
  nextError: string | null,
): boolean {
  return nextError !== null && nextError !== prevError;
}

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

export function isActivationKey(key: string): boolean {
  return key === "Enter" || key === " ";
}

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
