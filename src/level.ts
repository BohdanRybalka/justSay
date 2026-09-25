/** A microphone level in dBFS as a 0..1 fraction of the meter: -60 dB and
 *  below is empty, 0 dB is full. */
export function levelFromDb(db: number): number {
  return Math.max(0, Math.min(1, (db + 60) / 60));
}
