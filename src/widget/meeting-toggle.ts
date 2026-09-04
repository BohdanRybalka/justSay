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
 * It answers 409, 410 or 507 — the three codes the backend produces only once
 * nothing is being recorded and both devices are released — and the indicator
 * comes down too, because left in the general branch a 409 lit it forever: the
 * next click took the stop branch, got the same 409, and only reloading the
 * widget window cleared it. Or it fails for any other reason — an unreachable
 * backend, a refused connection — where nothing says the capture ended, so the
 * indicator stays up.
 *
 * Neither meeting call carries a budget (ADR 049, fourth amendment), so no
 * outcome here is an abandoned request: every failure this function sees was
 * observed.
 *
 * The three share a branch and not a message. 409 is a double click; 410 is a
 * call that ran and captured nothing, which is news rather than a mistimed
 * press, so it must not be described as already stopped; 507 is a call that
 * ran, captured audio and lost it on the way to disk, which is the worst of
 * the three and must not be described as either of the others. The backend
 * answers 507 rather than the 500 a failed write used to raise precisely so
 * this branch can tell it apart from a backend that never answered.
 *
 * A start has its own 409, and it means the opposite of everything above: the
 * backend refuses to start because it is already holding both devices. That is
 * the one start failure after which something *is* being recorded, so it puts
 * the indicator up rather than down. Reaching it means the widget had lost
 * track — a window reload whose status call failed leaves `meetingActive`
 * false while the call keeps recording — and hiding the indicator there both
 * broke ADR 040 obligation 2 and left the recording unstoppable, because the
 * next click would take the start branch again and get the same 409.
 *
 * The sentence a user reads is written here and the backend detail is appended
 * to it as its cause, which is why the 507 detail carries the write error alone
 * and not a second sentence of its own.
 */

import { ApiRequestError } from "../api";

export const DISCLOSURE_REQUIRED_MESSAGE =
  "Read the meeting-recording disclosure before recording a call.";

const DISCLOSURE_REQUIRED_STATUS = 403;
const ALREADY_STOPPED_STATUS = 409;
const ALREADY_RECORDING_STATUS = 409;
const NOTHING_CAPTURED_STATUS = 410;
const WRITE_FAILED_STATUS = 507;

const CAPTURE_OVER_STATUSES = [
  ALREADY_STOPPED_STATUS,
  NOTHING_CAPTURED_STATUS,
  WRITE_FAILED_STATUS,
];

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

function writeFailedMessage(error: unknown): string {
  return `The call ended but its recording could not be saved: ${describeFailure(error)}`;
}

function alreadyRecordingMessage(error: unknown): string {
  return `A call is already being recorded: ${describeFailure(error)}`;
}

function captureOverMessage(error: ApiRequestError): string {
  if (error.status === NOTHING_CAPTURED_STATUS) {
    return nothingCapturedMessage(error);
  }
  if (error.status === WRITE_FAILED_STATUS) {
    return writeFailedMessage(error);
  }
  return alreadyStoppedMessage(error);
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
      if (e instanceof ApiRequestError && CAPTURE_OVER_STATUSES.includes(e.status)) {
        actions.hideIndicator();
        await actions.setTrayRecording(false);
        actions.reportError(captureOverMessage(e));
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
    if (e instanceof ApiRequestError && e.status === ALREADY_RECORDING_STATUS) {
      actions.showIndicator();
      await actions.setTrayRecording(true);
      actions.reportError(alreadyRecordingMessage(e));
      return;
    }
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
