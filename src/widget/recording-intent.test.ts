import { describe, expect, it, vi } from "vitest";
import { createRecordingIntentQueue, type RecordingIntentActions } from "./recording-intent";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

/** The recorder the widget drives: `startRecording` and `stopAndProcess` both
 *  move the widget's state synchronously and only then await the backend, so
 *  the fakes below flip `recording` before their gate, not after it. */
function actions(recorder: { recording: boolean }, overrides: Partial<RecordingIntentActions> = {}) {
  const spies = {
    isRecording: vi.fn(() => recorder.recording),
    isBusy: vi.fn(() => false),
    startRecording: vi.fn(async () => {
      recorder.recording = true;
    }),
    stopRecording: vi.fn(async () => {
      recorder.recording = false;
    }),
    reportError: vi.fn(),
  };
  return Object.assign(spies, overrides);
}

describe("the recording intent queue", () => {
  it("stops the recording when a release arrives before the start has answered", async () => {
    const recorder = { recording: false };
    const started = deferred();
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        recorder.recording = true;
        await started.promise;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const pressed = queue.request("start");
    const released = queue.request("stop");
    started.resolve();
    await Promise.all([pressed, released]);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(deps.startRecording.mock.invocationCallOrder[0]).toBeLessThan(
      deps.stopRecording.mock.invocationCallOrder[0],
    );
    expect(recorder.recording).toBe(false);
  });

  it("stops the recording when a second click arrives before the start has answered", async () => {
    const recorder = { recording: false };
    const started = deferred();
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        recorder.recording = true;
        await started.promise;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const first = queue.request("toggle");
    const second = queue.request("toggle");
    started.resolve();
    await Promise.all([first, second]);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(recorder.recording).toBe(false);
  });

  it("does nothing on a stop that follows a failed start", async () => {
    const recorder = { recording: false };
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        throw new Error("backend unreachable");
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    await queue.request("start");
    await queue.request("stop");

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.stopRecording).not.toHaveBeenCalled();
  });

  it("coalesces two press/release pairs inside one unanswered start into one recording", async () => {
    const recorder = { recording: false };
    const started = deferred();
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        recorder.recording = true;
        await started.promise;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const burst = [
      queue.request("start"),
      queue.request("stop"),
      queue.request("start"),
      queue.request("stop"),
    ];
    started.resolve();
    await Promise.all(burst);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(recorder.recording).toBe(false);
  });

  it("applies a press that arrives while a dictation is being processed", async () => {
    const recorder = { recording: true };
    const processed = deferred();
    const deps = actions(recorder, {
      stopRecording: vi.fn(async () => {
        recorder.recording = false;
        await processed.promise;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const released = queue.request("stop");
    const pressed = queue.request("start");
    processed.resolve();
    await Promise.all([released, pressed]);

    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(recorder.recording).toBe(true);
  });

  it("reports a rejecting action once and still runs the next intent", async () => {
    const recorder = { recording: false };
    const deps = actions(recorder, {
      startRecording: vi
        .fn()
        .mockRejectedValueOnce(new Error("transition bug"))
        .mockImplementationOnce(async () => {
          recorder.recording = true;
        }),
    });
    const queue = createRecordingIntentQueue(deps);

    await queue.request("start");
    await queue.request("start");

    expect(deps.reportError).toHaveBeenCalledOnce();
    expect(deps.reportError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(deps.startRecording).toHaveBeenCalledTimes(2);
    expect(recorder.recording).toBe(true);
  });

  it("does nothing on a stop when nothing is recording", async () => {
    const recorder = { recording: false };
    const deps = actions(recorder);

    await createRecordingIntentQueue(deps).request("stop");

    expect(deps.stopRecording).not.toHaveBeenCalled();
    expect(deps.startRecording).not.toHaveBeenCalled();
    expect(deps.reportError).not.toHaveBeenCalled();
  });
  it("serves an intent that arrived while a failing action was in flight", async () => {
    const recorder = { recording: false };
    const started = deferred();
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        recorder.recording = true;
        await started.promise;
        throw new Error("the backend refused the start");
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const pressed = queue.request("start");
    const released = queue.request("stop");
    started.resolve();
    await Promise.all([pressed, released]);

    expect(deps.reportError).toHaveBeenCalledOnce();
    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(recorder.recording).toBe(false);
  });

  it("calls nothing further when a start that swallows its own failure leaves nothing recording", async () => {
    const recorder = { recording: false };
    const started = deferred();
    const deps = actions(recorder, {
      startRecording: vi.fn(async () => {
        await started.promise;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const pressed = queue.request("start");
    const released = queue.request("stop");
    started.resolve();
    await Promise.all([pressed, released]);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.stopRecording).not.toHaveBeenCalled();
    expect(deps.reportError).not.toHaveBeenCalled();
    expect(recorder.recording).toBe(false);
  });

  it("ignores a click that arrives while a dictation is being processed", async () => {
    const recorder = { recording: false };
    const deps = actions(recorder, { isBusy: vi.fn(() => true) });

    await createRecordingIntentQueue(deps).request("toggle");

    expect(deps.startRecording).not.toHaveBeenCalled();
    expect(deps.stopRecording).not.toHaveBeenCalled();
  });

  it("still applies a hotkey press that arrives while a dictation is being processed", async () => {
    const recorder = { recording: true };
    const processed = deferred();
    let busy = false;
    const deps = actions(recorder, {
      isBusy: vi.fn(() => busy),
      stopRecording: vi.fn(async () => {
        recorder.recording = false;
        busy = true;
        await processed.promise;
        busy = false;
      }),
    });
    const queue = createRecordingIntentQueue(deps);

    const released = queue.request("stop");
    const pressed = queue.request("start");
    processed.resolve();
    await Promise.all([released, pressed]);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(recorder.recording).toBe(true);
  });
});
