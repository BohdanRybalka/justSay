/**
 * The one decision behind starting and stopping a dictation: what happens to an
 * intent the user expresses while the previous transition is still in flight.
 *
 * It lives outside widget.ts because the old answer was "drop it", and dropping
 * it is JS-103: a push-to-talk tap short enough that `Released` arrives before
 * `POST /audio/start` has answered left the widget in `recording`, the duration
 * timer running and the microphone open until the user pressed the hotkey a
 * second time. A boolean guard cannot tell "this intent is redundant" from
 * "this intent is the one that ends the recording", and which of the two it is
 * only becomes knowable after the in-flight call settles.
 *
 * The answer here is that an intent is remembered as the latest desired state,
 * never dropped and never replayed as a sequence. Coalescing to the latest wish
 * is deliberate: replaying two press/release pairs would produce two dictations
 * from one burst of taps, the second of them empty. Its symmetric consequence
 * is deliberate too — a press arriving during processing is applied once
 * processing settles, so a held key starts a recording late instead of not at
 * all. A `"toggle"` is the exception and is dropped while the widget is busy:
 * a click is an instantaneous action rather than a sustained one, so queueing
 * it would start a whole new dictation the moment the previous one finished
 * processing — which is not what the click meant.
 *
 * "Never dropped" also covers the failing action: if the injected start or stop
 * rejects, the error is reported once and any intent that arrived while it was
 * in flight is still served. Throwing that intent away is JS-103 in its
 * original shape — a release lost because the start it raced never answered.
 */

export type RecordingIntent = "start" | "stop" | "toggle";

/** Everything the queue needs from the widget, so the decision can be driven
 *  without a DOM, a backend or a Tauri bridge. */
export interface RecordingIntentActions {
  isRecording(): boolean;
  isBusy(): boolean;
  startRecording(): Promise<unknown>;
  stopRecording(): Promise<unknown>;
  reportError(error: unknown): void;
}

export interface RecordingIntentQueue {
  request(intent: RecordingIntent): Promise<void>;
}

function desiredStateAfter(intent: RecordingIntent, current: boolean): boolean {
  if (intent === "toggle") return !current;
  return intent === "start";
}

export function createRecordingIntentQueue(
  actions: RecordingIntentActions,
): RecordingIntentQueue {
  let desired: boolean | null = null;
  let draining: Promise<void> | null = null;

  async function drain(): Promise<void> {
    while (desired !== null) {
      const target = desired;
      if (target === actions.isRecording()) {
        desired = null;
        return;
      }
      try {
        await (target ? actions.startRecording() : actions.stopRecording());
      } catch (e) {
        actions.reportError(e);
      }
      if (desired === target) desired = null;
    }
  }

  return {
    request(intent: RecordingIntent): Promise<void> {
      if (intent === "toggle" && actions.isBusy()) return Promise.resolve();
      desired = desiredStateAfter(intent, desired ?? actions.isRecording());
      if (draining) return draining;
      draining = drain().finally(() => {
        draining = null;
      });
      return draining;
    },
  };
}
