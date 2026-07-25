import { describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "../api";
import {
  DISCLOSURE_REQUIRED_MESSAGE,
  type MeetingToggleActions,
  runMeetingToggle,
} from "./meeting-toggle";

function actions(overrides: Partial<MeetingToggleActions> = {}) {
  const spies = {
    isRecording: vi.fn(() => false),
    startRecording: vi.fn(async () => ({})),
    stopRecording: vi.fn(async () => ({})),
    showIndicator: vi.fn(),
    hideIndicator: vi.fn(),
    setTrayRecording: vi.fn(async () => {}),
    openDisclosure: vi.fn(async () => {}),
    reportError: vi.fn(),
  };
  return Object.assign(spies, overrides);
}

describe("the meeting recording toggle", () => {
  it("raises the indicator and the tray state after a successful start", async () => {
    const deps = actions();

    await runMeetingToggle(deps);

    expect(deps.startRecording).toHaveBeenCalledOnce();
    expect(deps.showIndicator).toHaveBeenCalledOnce();
    expect(deps.setTrayRecording).toHaveBeenCalledWith(true);
    expect(deps.reportError).not.toHaveBeenCalled();
  });

  it("clears the indicator and the tray state after a successful stop", async () => {
    const deps = actions({ isRecording: vi.fn(() => true) });

    await runMeetingToggle(deps);

    expect(deps.stopRecording).toHaveBeenCalledOnce();
    expect(deps.startRecording).not.toHaveBeenCalled();
    expect(deps.hideIndicator).toHaveBeenCalledOnce();
    expect(deps.setTrayRecording).toHaveBeenCalledWith(false);
  });

  it("clears the indicator when starting fails, because nothing is recording", async () => {
    const deps = actions({
      startRecording: vi.fn(async () => {
        throw new Error("backend unreachable");
      }),
    });

    await runMeetingToggle(deps);

    expect(deps.hideIndicator).toHaveBeenCalledOnce();
    expect(deps.showIndicator).not.toHaveBeenCalled();
    expect(deps.setTrayRecording).toHaveBeenCalledWith(false);
    expect(deps.reportError).toHaveBeenCalledWith("backend unreachable");
  });

  it("keeps the indicator up when stopping fails, because the call is still being recorded", async () => {
    const deps = actions({
      isRecording: vi.fn(() => true),
      stopRecording: vi.fn(async () => {
        throw new ApiRequestError("HTTP 401", 401);
      }),
    });

    await runMeetingToggle(deps);

    expect(deps.hideIndicator).not.toHaveBeenCalled();
    expect(deps.setTrayRecording).not.toHaveBeenCalled();
    expect(deps.reportError).toHaveBeenCalledOnce();
    expect(deps.reportError.mock.calls[0][0]).toContain("still being recorded");
  });

  it("leaves the next click on the stop branch after a failed stop", async () => {
    let recording = true;
    const deps = actions({
      isRecording: vi.fn(() => recording),
      stopRecording: vi
        .fn()
        .mockRejectedValueOnce(new Error("timed out"))
        .mockResolvedValueOnce({}),
      hideIndicator: vi.fn(() => {
        recording = false;
      }),
    });

    await runMeetingToggle(deps);
    await runMeetingToggle(deps);

    expect(deps.stopRecording).toHaveBeenCalledTimes(2);
    expect(deps.startRecording).not.toHaveBeenCalled();
    expect(deps.hideIndicator).toHaveBeenCalledOnce();
    expect(deps.setTrayRecording).toHaveBeenCalledWith(false);
  });

  it("opens the disclosure when a start is refused with 403", async () => {
    const deps = actions({
      startRecording: vi.fn(async () => {
        throw new ApiRequestError("Disclosure not acknowledged", 403);
      }),
    });

    await runMeetingToggle(deps);

    expect(deps.openDisclosure).toHaveBeenCalledOnce();
    expect(deps.reportError).toHaveBeenCalledWith(DISCLOSURE_REQUIRED_MESSAGE);
    expect(deps.hideIndicator).toHaveBeenCalledOnce();
    expect(deps.setTrayRecording).toHaveBeenCalledWith(false);
  });
});
