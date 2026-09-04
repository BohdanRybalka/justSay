/**
 * The one answer to "a poll answered late, may it repaint?".
 *
 * Every periodic read in this app is started by a `setInterval` that does not
 * await it, so against a backend that accepts and then goes quiet a probe
 * outlives its own tick and several overlap. A boolean in-flight guard answers
 * that by *dropping* the later probe, which costs a whole budget of staleness:
 * the screen can sit a full `REQUEST_TIMEOUT_MS` past a tick before it learns
 * the state changed. A generation counter runs every probe and discards only
 * the answers a newer probe has already superseded, which is the same
 * protection without the dropped read.
 *
 * It lives at the root rather than in `settings/tabs/` because the widget polls
 * too, and two mechanisms for one problem is what the eighth review of PR #85
 * rejected.
 */
export function isStaleStatusResponse(requestToken: number, latestIssuedToken: number): boolean {
  return requestToken !== latestIssuedToken;
}
