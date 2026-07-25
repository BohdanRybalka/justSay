/**
 * The one decision behind the meeting-recording toggle: what the indicator and
 * the tray must say after the backend answers — or fails to.
 *
 * It lives outside widget.ts because a failed *start* and a failed *stop* mean
 * opposite things and the difference is not obvious enough to leave untested.
 * After a failed start nothing is being captured, so the indicator comes down.
 * After a failed stop the backend is most likely still recording, so the
 * indicator must stay up: ADR 040 obligation 2 forbids an indicator that
 * disappears while a recording continues, and clearing it would also make the
 * recording unstoppable — the next click would take the start branch and be
 * refused with 409, hiding the indicator again.
 */

import { ApiRequestError } from "../api";

export const DISCLOSURE_REQUIRED_MESSAGE =
  "Read the meeting-recording disclosure before recording a call.";

const DISCLOSURE_REQUIRED_STATUS = 403;

export function describeFailure(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function stopFailureMessage(error: unknown): string {
  return `The call is still being recorded — stopping it failed: ${describeFailure(error)}`;
}

/** Everything the toggle needs from the widget, so the decision can be driven
 *  without a DOM, a backend or a Tauri bridge. */
export interface MeetingToggleActions {
  isRecording(): boolean;
  startRecording(): Promise<unknown>;
  stopRecording(): Promise<unknown>;
  showIndicator(): void;
  hideIndicator(): void;
  setTrayRecording(active: boolean): Promise<void>;
  openDisclosure(): Promise<void>;
  reportError(message: string): void;
}

export async function runMeetingToggle(actions: MeetingToggleActions): Promise<void> {
  if (actions.isRecording()) {
    try {
      await actions.stopRecording();
    } catch (e) {
      actions.reportError(stopFailureMessage(e));
      return;
    }
    actions.hideIndicator();
    await actions.setTrayRecording(false);
    return;
  }

  try {
    await actions.startRecording();
  } catch (e) {
    actions.hideIndicator();
    await actions.setTrayRecording(false);
    if (e instanceof ApiRequestError && e.status === DISCLOSURE_REQUIRED_STATUS) {
      await actions.openDisclosure();
      actions.reportError(DISCLOSURE_REQUIRED_MESSAGE);
      return;
    }
    actions.reportError(describeFailure(e));
    return;
  }

  actions.showIndicator();
  await actions.setTrayRecording(true);
}
