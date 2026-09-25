/** A level in dBFS as a 0..1 fraction of the meter: -60 dB and below, or no
 *  level at all, is empty; 0 dB is full. */
export function levelFromDb(db: number | null): number {
  if (db === null) return 0;
  return Math.max(0, Math.min(1, (db + 60) / 60));
}
