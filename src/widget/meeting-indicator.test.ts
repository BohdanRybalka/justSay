// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import {
  MEETING_DEGRADED_CLASS,
  MEETING_DURATION_ID,
  MEETING_STATE_CLASS,
  renderMeetingIndicator,
} from "./meeting-indicator";

function widget(): { root: HTMLElement; dictation: HTMLElement; meeting: HTMLElement } {
  const root = document.createElement("div");
  root.className = "widget idle";
  const dictation = document.createElement("span");
  dictation.id = "widget-duration";
  const meeting = document.createElement("span");
  meeting.id = MEETING_DURATION_ID;
  root.append(dictation, meeting);
  return { root, dictation, meeting };
}

describe("the meeting recording indicator (ADR 040 obligation 2)", () => {
  it("marks the widget root and shows a duration while a meeting is recording", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 5, incident: null });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(meeting.textContent).not.toBe("");
  });

  it("marks the root degraded, without dropping the recording state, on an incident", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, {
      active: true,
      elapsedSeconds: 5,
      incident: "system_audio_ended",
    });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(root.classList.contains(MEETING_DEGRADED_CLASS)).toBe(true);
    expect(meeting.textContent).not.toBe("");
  });

  it("takes the degraded mark off again when the call ends", () => {
    const { root } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 5, incident: "storage_low" });
    renderMeetingIndicator(root, { active: false, elapsedSeconds: 0, incident: null });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(root.classList.contains(MEETING_DEGRADED_CLASS)).toBe(false);
  });

  it("carries neither the state nor a duration when nothing is recording", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 0, incident: null });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(meeting.textContent).toBe("");
  });

  it("disappears when the recording stops, after having been shown", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 61, incident: null });
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 61, incident: null });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(meeting.textContent).toBe("");
  });

  it("leaves the dictation state class alone in both directions", () => {
    const { root } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 1, incident: null });
    renderMeetingIndicator(root, { active: false, elapsedSeconds: 1, incident: null });

    expect(root.classList.contains("idle")).toBe(true);
  });

  it("advances the readout as the recording runs", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 1, incident: null });
    const first = meeting.textContent;
    renderMeetingIndicator(root, { active: true, elapsedSeconds: 75, incident: null });

    expect(meeting.textContent).not.toBe(first);
    expect(meeting.textContent).toBe("1:15");
  });

  it("survives the widget clearing dictation's counter, which is the defect", () => {
    const { root, dictation, meeting } = widget();
    renderMeetingIndicator(root, { active: true, elapsedSeconds: 75, incident: null });

    dictation.textContent = "";

    expect(meeting.textContent).toBe("1:15");
  });

  it("never writes dictation's counter, in either direction", () => {
    const { root, dictation } = widget();
    dictation.textContent = "0:07";

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 30, incident: null });
    expect(dictation.textContent).toBe("0:07");

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 30, incident: null });
    expect(dictation.textContent).toBe("0:07");
  });

  it("says so rather than going quiet when its readout is missing", () => {
    const root = document.createElement("div");
    const reported: unknown[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => reported.push(args);

    try {
      expect(() => renderMeetingIndicator(root, { active: true, elapsedSeconds: 1, incident: null })).not.toThrow();
    } finally {
      console.error = original;
    }

    expect(reported).toHaveLength(1);
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
  });
});
