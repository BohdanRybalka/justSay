// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { MEETING_STATE_CLASS, renderMeetingIndicator } from "./meeting-indicator";

function widget(): { root: HTMLElement; duration: HTMLElement } {
  const root = document.createElement("div");
  root.className = "widget idle";
  const duration = document.createElement("span");
  root.appendChild(duration);
  return { root, duration };
}

describe("the meeting recording indicator (ADR 040 obligation 2)", () => {
  it("marks the widget root and shows a duration while a meeting is recording", () => {
    const { root, duration } = widget();

    renderMeetingIndicator(root, duration, { active: true, elapsedSeconds: 5 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(duration.textContent).not.toBe("");
  });

  it("carries neither the state nor a duration when nothing is recording", () => {
    const { root, duration } = widget();

    renderMeetingIndicator(root, duration, { active: false, elapsedSeconds: 0 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(duration.textContent).toBe("");
  });

  it("disappears when the recording stops, after having been shown", () => {
    const { root, duration } = widget();

    renderMeetingIndicator(root, duration, { active: true, elapsedSeconds: 61 });
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);

    renderMeetingIndicator(root, duration, { active: false, elapsedSeconds: 61 });

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(duration.textContent).toBe("");
  });

  it("leaves the dictation state class alone in both directions", () => {
    const { root, duration } = widget();

    renderMeetingIndicator(root, duration, { active: true, elapsedSeconds: 1 });
    renderMeetingIndicator(root, duration, { active: false, elapsedSeconds: 1 });

    expect(root.classList.contains("idle")).toBe(true);
  });

  it("advances the readout as the recording runs", () => {
    const { root, duration } = widget();

    renderMeetingIndicator(root, duration, { active: true, elapsedSeconds: 1 });
    const first = duration.textContent;
    renderMeetingIndicator(root, duration, { active: true, elapsedSeconds: 75 });

    expect(duration.textContent).not.toBe(first);
    expect(duration.textContent).toBe("1:15");
  });
});
