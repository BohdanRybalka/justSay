import { describe, expect, it } from "vitest";
import type { MeetingStatus } from "../api";
import { CAPTURE_INCIDENTS } from "../contracts";
import { decideMeetingHealth, describeMeetingIncident } from "./meeting-health";

function status(overrides: Partial<MeetingStatus> = {}): MeetingStatus {
  return {
    is_recording: true,
    duration_seconds: 42,
    level_db: -30,
    system_endpoint: "Headset [Loopback]",
    system_level_db: -30,
    capture_incident: null,
    ...overrides,
  };
}

describe("what a polled meeting status tells the widget to do", () => {
  it("keeps the marker untouched while a healthy capture is running", () => {
    expect(decideMeetingHealth(status(), true)).toEqual({ kind: "keep" });
  });

  it("keeps out of the way when this window believes no meeting is running", () => {
    expect(decideMeetingHealth(status({ is_recording: false }), false)).toEqual({
      kind: "keep",
    });
  });

  it("ends the marker when the backend is no longer recording the call this window shows", () => {
    const action = decideMeetingHealth(status({ is_recording: false }), true);

    expect(action.kind).toBe("end");
    expect(action.kind === "end" && action.message).toMatch(/no longer being recorded/);
  });

  it("degrades rather than ends while the capture is still running with an incident", () => {
    for (const incident of CAPTURE_INCIDENTS) {
      const action = decideMeetingHealth(status({ capture_incident: incident }), true);

      expect(action.kind).toBe("degrade");
      expect(action.kind === "degrade" && action.incident).toBe(incident);
      expect(action.kind === "degrade" && action.message).toContain(
        describeMeetingIncident(incident),
      );
    }
  });

  it("still says something when the backend names an incident this build does not know", () => {
    const action = decideMeetingHealth(status({ capture_incident: "meteor_strike" }), true);

    expect(action.kind).toBe("degrade");
    expect(action.kind === "degrade" && action.message).toMatch(/something went wrong/);
  });

  it("gives every known incident its own sentence, so no two read the same", () => {
    const clauses = CAPTURE_INCIDENTS.map(describeMeetingIncident);

    expect(new Set(clauses).size).toBe(CAPTURE_INCIDENTS.length);
    expect(clauses).not.toContain(describeMeetingIncident("meteor_strike"));
  });
});
