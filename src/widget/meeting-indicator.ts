/**
 * The widget's meeting-recording indicator.
 *
 * ADR 040 obligation 2 requires a persistent, product-owned visible indicator
 * for the whole duration of a meeting recording, and ADR 041 records that the
 * macOS Core Audio tap produces no menu-bar indicator of its own — so this is
 * the only thing anyone can see. It lives here rather than inside widget.ts so
 * that "the indicator is showing" is assertable against a DOM without booting
 * the whole widget.
 */

import { formatElapsedClock } from "../format";

export const MEETING_STATE_CLASS = "meeting";

export interface MeetingIndicatorState {
  active: boolean;
  elapsedSeconds: number;
}

export function renderMeetingIndicator(
  root: HTMLElement,
  durationEl: HTMLElement,
  state: MeetingIndicatorState,
): void {
  if (state.active) {
    root.classList.add(MEETING_STATE_CLASS);
    durationEl.textContent = formatElapsedClock(state.elapsedSeconds);
    return;
  }
  root.classList.remove(MEETING_STATE_CLASS);
  durationEl.textContent = "";
}
