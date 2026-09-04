import { describe, expect, it, vi } from "vitest";
import { ApiAuthError, ApiRequestError, REQUEST_TIMEOUT_MS } from "../api";
import { TimedOutError } from "../timeout";
import {
  createAbandonedStartCleanup,
  indicatorAfterAbandonedMeetingStart,
  readRecordingTruth,
  type RecordingSnapshot,
} from "./abandoned-request";

describe("readRecordingTruth", () => {
  it("reports a live recording with the elapsed time the backend gives", async () => {
    const read = vi.fn(async (): Promise<RecordingSnapshot> => ({
      is_recording: true,
      duration_seconds: 12.5,
    }));

    expect(await readRecordingTruth(read)).toEqual({ kind: "recording", elapsedSeconds: 12.5 });
    expect(read).toHaveBeenCalledOnce();
  });

  it("reports idle when the backend says nothing is being recorded", async () => {
    const read = async (): Promise<RecordingSnapshot> => ({
      is_recording: false,
      duration_seconds: 0,
    });

    expect(await readRecordingTruth(read)).toEqual({ kind: "idle" });
  });

  it("reports unknown rather than throwing when the status read fails as well", async () => {
    const read = async (): Promise<RecordingSnapshot> => {
      throw new Error("the backend did not answer /audio/status within 15 seconds");
    };

    expect(await readRecordingTruth(read)).toEqual({ kind: "unknown" });
  });

  it("asks once and does not poll", async () => {
    const read = vi.fn(async (): Promise<RecordingSnapshot> => ({
      is_recording: false,
      duration_seconds: 0,
    }));

    await readRecordingTruth(read);

    expect(read).toHaveBeenCalledTimes(1);
  });
});

describe("indicatorAfterAbandonedMeetingStart", () => {
  it("shows the indicator for a recording the backend really holds", () => {
    expect(indicatorAfterAbandonedMeetingStart({ kind: "recording", elapsedSeconds: 3 })).toBe(
      "show",
    );
  });

  it("hides it only when the backend positively reports no recording", () => {
    expect(indicatorAfterAbandonedMeetingStart({ kind: "idle" })).toBe("hide");
  });

  it("shows it when the truth is unknown, because a false negative has no recovery", () => {
    expect(indicatorAfterAbandonedMeetingStart({ kind: "unknown" })).toBe("show");
  });
});

describe("the stop an abandoned start owes", () => {
  function cleanup(overrides: Partial<{ stopRecording: () => Promise<unknown>; isBusy: () => boolean }> = {}) {
    const deps = {
      stopRecording: vi.fn(async () => ({ filename: "rec.wav" })),
      isBusy: vi.fn(() => false),
      ...overrides,
    };
    return { deps, subject: createAbandonedStartCleanup(deps) };
  }

  it("owes nothing until a start has been abandoned", async () => {
    const { deps, subject } = cleanup();

    expect(await subject.settle(true)).toBe("nothing-owed");
    expect(deps.stopRecording).not.toHaveBeenCalled();
  });

  it("issues exactly one stop on the first poll that reaches a live backend", async () => {
    const { deps, subject } = cleanup();
    subject.owe();

    expect(await subject.settle(true)).toBe("settled");
    expect(await subject.settle(true)).toBe("nothing-owed");
    expect(deps.stopRecording).toHaveBeenCalledOnce();
  });

  it("holds the debt while the backend is unreachable and pays it when it answers", async () => {
    const { deps, subject } = cleanup();
    subject.owe();

    for (let poll = 0; poll < 6; poll += 1) {
      expect(await subject.settle(false)).toBe("deferred");
    }
    expect(deps.stopRecording).not.toHaveBeenCalled();

    expect(await subject.settle(true)).toBe("settled");
    expect(deps.stopRecording).toHaveBeenCalledOnce();
  });

  it("never touches a recording the widget is using", async () => {
    const { deps, subject } = cleanup({ isBusy: () => true });
    subject.owe();

    expect(await subject.settle(true)).toBe("deferred");
    expect(deps.stopRecording).not.toHaveBeenCalled();
  });

  it("keeps a single stop in flight even though the poll is faster than the budget", async () => {
    let release!: () => void;
    const stopRecording = vi.fn(
      () => new Promise<unknown>((resolve) => (release = () => resolve({}))),
    );
    const { subject } = cleanup({ stopRecording });
    subject.owe();

    const first = subject.settle(true);
    expect(await subject.settle(true)).toBe("deferred");
    expect(await subject.settle(true)).toBe("deferred");
    expect(stopRecording).toHaveBeenCalledOnce();

    release();
    expect(await first).toBe("settled");
  });

  it("treats a refusal as the answer it was missing", async () => {
    const stopRecording = vi.fn(async () => {
      throw new ApiRequestError("Not recording", 409);
    });
    const { subject } = cleanup({ stopRecording });
    subject.owe();

    expect(await subject.settle(true)).toBe("settled");
    expect(await subject.settle(true)).toBe("nothing-owed");
    expect(stopRecording).toHaveBeenCalledOnce();
  });

  it("treats a 401 as an answer too, because the backend saw the request", async () => {
    const stopRecording = vi.fn(async () => {
      throw new ApiAuthError("Unauthorized", { kind: "ok" });
    });
    const { subject } = cleanup({ stopRecording });
    subject.owe();

    expect(await subject.settle(true)).toBe("settled");
    expect(await subject.settle(true)).toBe("nothing-owed");
  });

  it("keeps the debt when the stop runs out of its own budget", async () => {
    const stopRecording = vi.fn(async () => {
      throw new TimedOutError(REQUEST_TIMEOUT_MS, "/audio/stop");
    });
    const { subject } = cleanup({ stopRecording });
    subject.owe();

    expect(await subject.settle(true)).toBe("deferred");
    expect(await subject.settle(true)).toBe("deferred");
    expect(stopRecording).toHaveBeenCalledTimes(2);
  });
});
