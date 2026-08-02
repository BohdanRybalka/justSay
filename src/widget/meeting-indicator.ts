/**
 * The widget's meeting-recording indicator.
 *
 * ADR 040 obligation 2 requires a persistent, product-owned visible indicator
 * for the whole duration of a meeting recording, and ADR 041 records that the
 * macOS Core Audio tap produces no menu-bar indicator of its own — so this is
 * the only thing anyone can see. It lives here rather than inside widget.ts so
 * that "the indicator is showing" is assertable against a DOM without booting
 * the whole widget.
 *
 * It reads its own readout out of the root rather than accepting one, because
 * accepting one is what let it share dictation's `#widget-duration`: every
 * branch of the widget's `setState` writes that node, so any state change
 * blanked a running meeting timer until the next tick. A marker that blinks is
 * weaker evidence than one that does not, and this marker is what the consent
 * story rests on.
 */

import { formatElapsedClock } from "../format";

export const MEETING_STATE_CLASS = "meeting";

export const MEETING_DURATION_ID = "widget-meeting-duration";

export interface MeetingIndicatorState {
  active: boolean;
  elapsedSeconds: number;
}

export function renderMeetingIndicator(root: HTMLElement, state: MeetingIndicatorState): void {
  const readout = root.querySelector<HTMLElement>(`#${MEETING_DURATION_ID}`);

  if (state.active) {
    root.classList.add(MEETING_STATE_CLASS);
    if (readout) readout.textContent = formatElapsedClock(state.elapsedSeconds);
    return;
  }
  root.classList.remove(MEETING_STATE_CLASS);
  if (readout) readout.textContent = "";
}
