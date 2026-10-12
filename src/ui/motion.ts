/**
 * Motion that scripts start, which the stylesheet's reduced-motion block cannot
 * reach: a script that staggers its own timers asks here first.
 */

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

/** True while the OS asks for less motion. Read at the moment of use, so a change applies to the next draw. */
export function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(REDUCED_MOTION_QUERY).matches;
}
