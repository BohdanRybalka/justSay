/**
 * Duration formatting, in one place.
 *
 * Three shapes existed as three private functions in three modules, two of them
 * called `formatDuration` with the same signature and incompatible output, and
 * two of them writing to the same DOM element. Each is named here for what it
 * produces, so picking the wrong one is a visible mistake rather than a silent
 * change of format.
 */

/** `m:ss`, counting up for as long as a dictation or a meeting recording runs. */
export function formatElapsedClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

/** `h m` / `m` — a total, read at a glance rather than watched. */
export function formatCoarseDuration(seconds: number): string {
  const minutes = wholeMinutes(seconds);
  const hours = Math.floor(minutes / 60);
  return hours > 0 ? `${hours} h ${minutes % 60} m` : `${minutes} m`;
}

/** `h:mm` — a long total set beside another one. */
export function formatHoursClock(seconds: number): string {
  const minutes = wholeMinutes(seconds);
  return `${Math.floor(minutes / 60)}:${(minutes % 60).toString().padStart(2, "0")}`;
}

/** The nearest whole minute; absent and negative totals are zero. */
export function wholeMinutes(seconds: number): number {
  return seconds > 0 ? Math.round(seconds / 60) : 0;
}
