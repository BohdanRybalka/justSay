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
 *
 * A stop has three outcomes here. It succeeds, and the indicator comes down.
 * It is refused with 409 or 410 — the two codes the backend answers only when
 * nothing is being recorded and both devices are released — and the indicator
 * comes down too, because left in the general branch a 409 lit it forever: the
 * next click took the stop branch, got the same 409, and only reloading the
 * widget window cleared it. Or it fails for any other reason, and the
 * indicator stays up.
 *
 * The two refusals share a branch and not a message. 409 is a double click;
 * 410 is a call that ran and captured nothing, which is news rather than a
 * mistimed press, so it must not be described as already stopped.
 */

import { ApiRequestError } from "../api";

export const DISCLOSURE_REQUIRED_MESSAGE =
  "Read the meeting-recording disclosure before recording a call.";

const DISCLOSURE_REQUIRED_STATUS = 403;
const ALREADY_STOPPED_STATUS = 409;
const NOTHING_CAPTURED_STATUS = 410;

function describeFailure(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function stopFailureMessage(error: unknown): string {
  return `The call is still being recorded — stopping it failed: ${describeFailure(error)}`;
}

function alreadyStoppedMessage(error: unknown): string {
  return `The call was already stopped: ${describeFailure(error)}`;
}

function nothingCapturedMessage(error: unknown): string {
  return `The call ended with no audio and nothing was saved: ${describeFailure(error)}`;
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
      if (
        e instanceof ApiRequestError &&
        (e.status === ALREADY_STOPPED_STATUS || e.status === NOTHING_CAPTURED_STATUS)
      ) {
        actions.hideIndicator();
        await actions.setTrayRecording(false);
        actions.reportError(
          e.status === NOTHING_CAPTURED_STATUS
            ? nothingCapturedMessage(e)
            : alreadyStoppedMessage(e)
        );
        return;
      }
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
