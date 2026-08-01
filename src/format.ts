/**
 * Duration formatting, in one place.
 *
 * Three shapes existed as three private functions in three modules, two of them
 * called `formatDuration` with the same signature and incompatible output, and
 * two of them writing to the same DOM element. Each is named here for what it
 * produces, so picking the wrong one is a visible mistake rather than a silent
 * change of format.
 */

/** `m:ss.d` while counting, `s.ds` under a minute — the dictation stopwatch. */
export function formatStopwatch(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);
  if (minutes > 0) return `${minutes}:${remainder.toString().padStart(2, "0")}.${tenths}`;
  return `${remainder}.${tenths}s`;
}

/** `m:ss`, counting up for as long as a meeting recording runs. */
export function formatElapsedClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

/** `h m` / `m s` / `s` — a total, read at a glance rather than watched. */
export function formatCoarseDuration(seconds: number): string {
  if (!seconds || seconds < 0) return "0 m";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = Math.floor(seconds % 60);
  if (hours > 0) return `${hours} h ${minutes} m`;
  if (minutes > 0) return `${minutes} m ${remainder} s`;
  return `${remainder} s`;
}
