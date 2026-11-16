import { describe, expect, it } from "vitest";
import { actionLabel, failureText, spokenLabel, stepFor, UNREAD_RING_STATE, type RingAction, type RingState } from "./ring-actions";

function read(overrides: Partial<NonNullable<RingState["settings"]>> = {}, meetingRecording = false): RingState {
  return {
    settings: {
      language: "uk",
      previous_language: "en",
      meeting_consent_acknowledged: true,
      shortcut: "Ctrl+Alt+KeyV",
      ...overrides,
    },
    meetingRecording,
  };
}

function text(action: RingAction, state: RingState): string {
  return spokenLabel(actionLabel(action, state, "windows"));
}

describe("the button labels", () => {
  it("name the language in use", () => {
    expect(text("language", read({ language: "uk" }))).toBe("Language · Ukrainian");
    expect(text("language", read({ language: "en" }))).toBe("Language · English");
  });

  it("say only Language before the settings have been read", () => {
    expect(text("language", UNREAD_RING_STATE)).toBe("Language");
  });

  it("offer to stop the meeting while one is being recorded", () => {
    expect(text("meeting", read())).toBe("Record a meeting");
    expect(text("meeting", read({}, true))).toBe("Stop the meeting");
  });

  it("offer the mic with the user's own shortcut, written for their system", () => {
    expect(actionLabel("dictate", read(), "windows")).toEqual({ text: "Dictate", shortcut: "Ctrl + Alt + V" });
    expect(actionLabel("dictate", read({ shortcut: "Super+Shift+KeyD" }), "mac")).toEqual({
      text: "Dictate",
      shortcut: "⇧⌘D",
    });
    expect(text("dictate", read())).toBe("Dictate · Ctrl + Alt + V");
  });

  it("say only Dictate before the settings have been read", () => {
    expect(actionLabel("dictate", UNREAD_RING_STATE, "windows")).toEqual({ text: "Dictate", shortcut: null });
    expect(text("dictate", UNREAD_RING_STATE)).toBe("Dictate");
  });

  it("never name the service that transcribes", () => {
    for (const action of ["meeting", "file", "dictate", "language", "settings"] as const) {
      expect(text(action, read())).not.toMatch(/groq|gemini|whisper|cloud|local/i);
    }
  });
});

describe("what a button does", () => {
  it("starts or stops a meeting once the disclosure is acknowledged", () => {
    expect(stepFor("meeting", read())).toEqual({ kind: "toggle-meeting" });
  });

  it("opens the Meetings page until the disclosure is acknowledged", () => {
    const opened = { kind: "open-panel", target: { panel: "meetings" } };

    expect(stepFor("meeting", read({ meeting_consent_acknowledged: false }))).toEqual(opened);
  });

  it("stops a running meeting even when the disclosure reads as not acknowledged", () => {
    expect(stepFor("meeting", read({ meeting_consent_acknowledged: false }, true))).toEqual({
      kind: "toggle-meeting",
    });
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

  it("hands the mic to the widget, whatever was read", () => {
    expect(stepFor("dictate", read())).toEqual({ kind: "toggle-dictation" });
    expect(stepFor("dictate", UNREAD_RING_STATE)).toEqual({ kind: "toggle-dictation" });
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

describe("what a failed button says", () => {
  it("names the language it could not switch to", () => {
    expect(failureText({ kind: "switch-language", language: "en" }, read())).toBe(
      "Couldn't switch to English — try again.",
    );
  });

  it("says whether the meeting failed to start or to stop", () => {
    const toggle = { kind: "toggle-meeting" } as const;

    expect(failureText(toggle, read())).toBe("Couldn't start the meeting — try again.");
    expect(failureText(toggle, read({}, true))).toBe("Couldn't stop the meeting — try again.");
  });

  it("says the dictation did not start", () => {
    expect(failureText({ kind: "toggle-dictation" }, read())).toBe("Couldn't start dictating — try again.");
  });

  it("names the file picker and the main window", () => {
    expect(failureText({ kind: "pick-file" }, read())).toBe("Couldn't open the file picker — try again.");
    expect(failureText({ kind: "open-panel", target: { panel: "settings" } }, read())).toBe(
      "Couldn't open JustSay — try again.",
    );
  });
});
