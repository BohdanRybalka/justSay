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
