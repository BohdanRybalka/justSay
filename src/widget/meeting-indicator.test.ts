// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import {
  MEETING_LOST_CLASS,
  MEETING_SLOT_CLASS,
  MEETING_STATE_CLASS,
  MIC_BASE_HEIGHTS,
  SYSTEM_BASE_HEIGHTS,
  incidentSide,
  renderMeetingIndicator,
  type MeetingIndicatorState,
} from "./meeting-indicator";
import { renderPill } from "./pill";

function widget(): HTMLElement {
  const root = document.createElement("div");
  root.className = "pill pill--rest";
  root.innerHTML = `<span class="pill-content"></span><span class="${MEETING_SLOT_CLASS}"></span>`;
  return root;
}

function running(overrides: Partial<MeetingIndicatorState> = {}): MeetingIndicatorState {
  return { active: true, elapsedSeconds: 5, incident: null, mic: 0, system: 0, ...overrides };
}

const STOPPED: MeetingIndicatorState = {
  active: false,
  elapsedSeconds: 0,
  incident: null,
  mic: 0,
  system: 0,
};

function readout(root: HTMLElement): string | null | undefined {
  return root.querySelector(".pill-meeting .pill-readout")?.textContent;
}

function meter(root: HTMLElement, side: "mic" | "system"): HTMLElement {
  return root.querySelector<HTMLElement>(`.pill-meter[data-side="${side}"]`)!;
}

function barHeights(root: HTMLElement, side: "mic" | "system"): number[] {
  return [...meter(root, side).querySelectorAll<HTMLElement>(".pill-meter-bars i")].map((bar) =>
    parseFloat(bar.style.height),
  );
}

function markLost(root: HTMLElement): boolean {
  return root.querySelector(".pill-meeting-mark")!.classList.contains(MEETING_LOST_CLASS);
}

describe("the meeting recording indicator (ADR 040 obligation 2)", () => {
  it("marks the widget root and shows a duration while a meeting is recording", () => {
    const root = widget();

    renderMeetingIndicator(root, running({ elapsedSeconds: 75 }));

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(readout(root)).toBe("1:15");
  });

  it("offers a stop button while a meeting is recording", () => {
    const root = widget();

    renderMeetingIndicator(root, running());

    expect(root.querySelector(".pill-meeting button.pill-stop")).not.toBeNull();
  });

  it("carries neither the state nor a duration when nothing is recording", () => {
    const root = widget();

    renderMeetingIndicator(root, STOPPED);

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(readout(root) ?? "").toBe("");
  });

  it("disappears when the recording stops, after having been shown", () => {
    const root = widget();
    renderMeetingIndicator(root, running({ incident: "storage_low", mic: 1, system: 1 }));

    renderMeetingIndicator(root, STOPPED);

    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(false);
    expect(readout(root)).toBe("");
    expect(markLost(root)).toBe(false);
  });

  it("leaves the dictation state classes alone in both directions", () => {
    const root = widget();

    renderMeetingIndicator(root, running());
    renderMeetingIndicator(root, STOPPED);

    expect(root.classList.contains("pill--rest")).toBe(true);
  });

  it("keeps its readout through every dictation repaint", () => {
    const root = widget();
    renderMeetingIndicator(root, running({ elapsedSeconds: 75 }));

    renderPill(root, { kind: "alert", label: "No connection" });
    renderPill(root, { kind: "rest", hint: "Ctrl + Alt + V" });

    expect(readout(root)).toBe("1:15");
  });

  it("moves each side's bars with that side's level only", () => {
    const root = widget();

    renderMeetingIndicator(root, running({ mic: 1, system: 0 }));

    expect(barHeights(root, "mic")).toEqual([...MIC_BASE_HEIGHTS]);
    expect(Math.max(...barHeights(root, "system"))).toBeLessThan(Math.min(...SYSTEM_BASE_HEIGHTS));
  });

  it("says so rather than going quiet when its slot is missing", () => {
    const root = document.createElement("div");
    const reported: unknown[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => reported.push(args);

    try {
      expect(() => renderMeetingIndicator(root, running())).not.toThrow();
    } finally {
      console.error = original;
    }

    expect(reported).toHaveLength(1);
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
  });
});

describe("an incident on a running meeting", () => {
  it.each([
    ["microphone_stalled", "mic"],
    ["system_audio_ended", "system"],
    ["storage_failed", "recording"],
    ["storage_low", "recording"],
    ["storage_backlog", "recording"],
    ["something_new", "recording"],
  ] as const)("puts %s on the %s", (incident, side) => {
    expect(incidentSide(incident)).toBe(side);
  });

  it("turns the lost side's icon into the triangle and flattens its bars", () => {
    const root = widget();

    renderMeetingIndicator(root, running({ incident: "system_audio_ended", mic: 1, system: 1 }));

    expect(meter(root, "system").classList.contains(MEETING_LOST_CLASS)).toBe(true);
    expect(Math.max(...barHeights(root, "system"))).toBeLessThan(Math.min(...SYSTEM_BASE_HEIGHTS));
    expect(meter(root, "mic").classList.contains(MEETING_LOST_CLASS)).toBe(false);
    expect(barHeights(root, "mic")).toEqual([...MIC_BASE_HEIGHTS]);
    expect(markLost(root)).toBe(false);
  });

  it("turns the dot into the triangle for a storage incident and keeps both meters", () => {
    const root = widget();

    renderMeetingIndicator(root, running({ incident: "storage_failed", mic: 1, system: 1 }));

    expect(markLost(root)).toBe(true);
    expect(root.classList.contains(MEETING_STATE_CLASS)).toBe(true);
    expect(barHeights(root, "mic")).toEqual([...MIC_BASE_HEIGHTS]);
    expect(barHeights(root, "system")).toEqual([...SYSTEM_BASE_HEIGHTS]);
  });
});
