// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import {
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

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 5 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(meeting.textContent).not.toBe("");
  });

  it("carries neither the state nor a duration when nothing is recording", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 0 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(meeting.textContent).toBe("");
  });

  it("disappears when the recording stops, after having been shown", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 61 });
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 61 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(meeting.textContent).toBe("");
  });

  it("leaves the dictation state class alone in both directions", () => {
    const { root } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 1 });
    renderMeetingIndicator(root, { active: false, elapsedSeconds: 1 });

    expect(root.classList.contains("idle")).toBe(true);
  });

  it("advances the readout as the recording runs", () => {
    const { root, meeting } = widget();

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 1 });
    const first = meeting.textContent;
    renderMeetingIndicator(root, { active: true, elapsedSeconds: 75 });

    expect(meeting.textContent).not.toBe(first);
    expect(meeting.textContent).toBe("1:15");
  });

  it("survives the widget clearing dictation's counter, which is the defect", () => {
    const { root, dictation, meeting } = widget();
    renderMeetingIndicator(root, { active: true, elapsedSeconds: 75 });

    dictation.textContent = "";

    expect(meeting.textContent).toBe("1:15");
  });

  it("never writes dictation's counter, in either direction", () => {
    const { root, dictation } = widget();
    dictation.textContent = "0:07";

    renderMeetingIndicator(root, { active: true, elapsedSeconds: 30 });
    expect(dictation.textContent).toBe("0:07");

    renderMeetingIndicator(root, { active: false, elapsedSeconds: 30 });
    expect(dictation.textContent).toBe("0:07");
  });

  it("does not throw when the root has no readout of its own", () => {
    const root = document.createElement("div");

    expect(() => renderMeetingIndicator(root, { active: true, elapsedSeconds: 1 })).not.toThrow();
  });
});
