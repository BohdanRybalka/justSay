import { describe, expect, it, vi } from "vitest";
import {
  indicatorAfterAbandonedMeetingStart,
  readRecordingTruth,
  stateAfterAbandonedStart,
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

describe("stateAfterAbandonedStart", () => {
  it("adopts a recording the backend really holds", () => {
    expect(stateAfterAbandonedStart({ kind: "recording", elapsedSeconds: 3 })).toBe("recording");
  });

  it("fails the start when the backend is idle", () => {
    expect(stateAfterAbandonedStart({ kind: "idle" })).toBe("error");
  });

  it("fails the start when the truth is unknown, rather than claiming a capture", () => {
    expect(stateAfterAbandonedStart({ kind: "unknown" })).toBe("error");
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
