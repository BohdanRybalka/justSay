import { describe, expect, it } from "vitest";
import { petalLabel, stepFor, UNREAD_RING_STATE, type RingState } from "./ring-actions";

function read(overrides: Partial<NonNullable<RingState["settings"]>> = {}, meetingRecording = false): RingState {
  return {
    settings: {
      language: "uk",
      previous_language: "en",
      meeting_consent_acknowledged: true,
      meetings_enabled: true,
      ...overrides,
    },
    meetingRecording,
  };
}

describe("the petal labels", () => {
  it("name the language in use", () => {
    expect(petalLabel("language", read({ language: "uk" }))).toBe("Language · Ukrainian");
    expect(petalLabel("language", read({ language: "en" }))).toBe("Language · English");
  });

  it("say only Language before the settings have been read", () => {
    expect(petalLabel("language", UNREAD_RING_STATE)).toBe("Language");
  });

  it("offer to stop the meeting while one is being recorded", () => {
    expect(petalLabel("meeting", read())).toBe("Record a meeting");
    expect(petalLabel("meeting", read({}, true))).toBe("Stop the meeting");
  });

  it("never name the service that transcribes", () => {
    for (const action of ["meeting", "file", "language", "settings"] as const) {
      expect(petalLabel(action, read())).not.toMatch(/groq|gemini|whisper|cloud|local/i);
    }
  });
});

describe("what a petal does", () => {
  it("starts or stops a meeting while meetings are on", () => {
    expect(stepFor("meeting", read())).toEqual({ kind: "toggle-meeting" });
  });

  it("opens the Meetings page while meetings are off", () => {
    const opened = { kind: "open-panel", target: { panel: "meetings" } };

    expect(stepFor("meeting", read({ meetings_enabled: false }))).toEqual(opened);
    expect(stepFor("meeting", read({ meeting_consent_acknowledged: false }))).toEqual(opened);
  });

  it("stops a running meeting even when meetings read as off", () => {
    expect(stepFor("meeting", read({ meetings_enabled: false }, true))).toEqual({ kind: "toggle-meeting" });
  });

  it("leaves the meeting to the widget when the settings could not be read", () => {
    expect(stepFor("meeting", UNREAD_RING_STATE)).toEqual({ kind: "toggle-meeting" });
  });

  it("swaps to the language used before, and back", () => {
    expect(stepFor("language", read({ language: "uk", previous_language: "en" }))).toEqual({
      kind: "switch-language",
      language: "en",
    });
    expect(stepFor("language", read({ language: "en", previous_language: "uk" }))).toEqual({
      kind: "switch-language",
      language: "uk",
    });
  });

  it("opens Dictation to pick a language when none was used before", () => {
    const opened = { kind: "open-panel", target: { panel: "dictation" } };

    expect(stepFor("language", read({ previous_language: "" }))).toEqual(opened);
    expect(stepFor("language", read({ language: "uk", previous_language: "uk" }))).toEqual(opened);
    expect(stepFor("language", UNREAD_RING_STATE)).toEqual(opened);
  });

  it("asks the shell for a file, whatever was read", () => {
    expect(stepFor("file", read())).toEqual({ kind: "pick-file" });
    expect(stepFor("file", UNREAD_RING_STATE)).toEqual({ kind: "pick-file" });
  });

  it("opens Settings for Settings", () => {
    expect(stepFor("settings", read())).toEqual({
      kind: "open-panel",
      target: { panel: "settings" },
    });
  });
});
