import { describe, expect, it } from "vitest";
import {
  DEFAULT_SHORTCUT,
  acceleratorFromKeyEvent,
  detectShortcutPlatform,
  formatAccelerator,
  modifierHint,
  shortcutFailureMessage,
  shouldReapplyShortcut,
} from "./accelerator";

function keyEvent(
  overrides: Partial<Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "altKey" | "shiftKey" | "metaKey">>,
): Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "altKey" | "shiftKey" | "metaKey"> {
  return {
    key: "v",
    code: "KeyV",
    ctrlKey: false,
    altKey: false,
    shiftKey: false,
    metaKey: false,
    ...overrides,
  };
}

describe("formatAccelerator", () => {
  it("renders Apple glyphs in Control-Option-Shift-Command order on mac", () => {
    expect(formatAccelerator("Super+Alt+KeyV", "mac")).toBe("⌥⌘V");
    expect(formatAccelerator("Ctrl+Shift+Digit1", "mac")).toBe("⌃⇧1");
  });

  it("renders the stored default byte-identically to the previous Windows label", () => {
    expect(formatAccelerator(DEFAULT_SHORTCUT, "windows")).toBe("Ctrl + Alt + V");
    expect(formatAccelerator("Ctrl+Alt+KeyV", "windows")).toBe("Ctrl + Alt + V");
  });

  it("names Super as Win on Windows", () => {
    expect(formatAccelerator("Super+Shift+KeyA", "windows")).toBe("Win + Shift + A");
    expect(formatAccelerator("Super+Alt+KeyV", "mac")).toBe("⌥⌘V");
  });

  it("leaves key codes that are not Key/Digit prefixed alone", () => {
    expect(formatAccelerator("Ctrl+Backquote", "windows")).toBe("Ctrl + Backquote");
    expect(formatAccelerator("Ctrl+ArrowUp", "windows")).toBe("Ctrl + ArrowUp");
    expect(formatAccelerator("Ctrl+F5", "windows")).toBe("Ctrl + F5");
    expect(formatAccelerator("Ctrl+Numpad1", "windows")).toBe("Ctrl + Numpad1");
  });

  it("strips Key and Digit only at the start of the code", () => {
    expect(formatAccelerator("Ctrl+Digit1", "windows")).toBe("Ctrl + 1");
    expect(formatAccelerator("Ctrl+TurnKey", "windows")).toBe("Ctrl + TurnKey");
    expect(formatAccelerator("Ctrl+TwoDigit", "windows")).toBe("Ctrl + TwoDigit");
  });

  it("renders all four modifiers together", () => {
    expect(formatAccelerator("Super+Ctrl+Alt+Shift+KeyK", "mac")).toBe("⌃⌥⇧⌘K");
    expect(formatAccelerator("Super+Ctrl+Alt+Shift+KeyK", "windows")).toBe("Win + Ctrl + Alt + Shift + K");
  });
});

describe("acceleratorFromKeyEvent", () => {
  it("emits only platform-neutral tokens for a Command+Option+V press", () => {
    const result = acceleratorFromKeyEvent(keyEvent({ metaKey: true, altKey: true }));

    expect(result).toEqual({ ok: true, accelerator: "Super+Alt+KeyV" });
  });

  it("never emits Command, Cmd, Option or an Apple glyph for any modifier combination", () => {
    const banned = /Command|Cmd|Option|⌘|⌥|⌃|⇧/;

    for (const ctrlKey of [false, true]) {
      for (const altKey of [false, true]) {
        for (const shiftKey of [false, true]) {
          for (const metaKey of [false, true]) {
            const result = acceleratorFromKeyEvent(
              keyEvent({ ctrlKey, altKey, shiftKey, metaKey }),
            );
            if (!result.ok) continue;
            expect(result.accelerator).not.toMatch(banned);
            const tokens = result.accelerator.split("+");
            for (const token of tokens.slice(0, -1)) {
              expect(["Ctrl", "Alt", "Shift", "Super"]).toContain(token);
            }
            expect(tokens[tokens.length - 1]).toBe("KeyV");
          }
        }
      }
    }
  });

  it("keeps waiting while only a modifier is held down", () => {
    for (const key of ["Control", "Shift", "Alt", "Meta"]) {
      expect(acceleratorFromKeyEvent(keyEvent({ key, ctrlKey: true }))).toEqual({
        ok: false,
        reason: "pending",
      });
    }
  });

  it("rejects a bare key with no modifier", () => {
    expect(acceleratorFromKeyEvent(keyEvent({}))).toEqual({ ok: false, reason: "no-modifier" });
  });

  it("reproduces the stored default from a Ctrl+Alt+V press", () => {
    expect(acceleratorFromKeyEvent(keyEvent({ ctrlKey: true, altKey: true }))).toEqual({
      ok: true,
      accelerator: DEFAULT_SHORTCUT,
    });
  });
});

describe("detectShortcutPlatform", () => {
  it("reads mac from navigator.platform", () => {
    expect(detectShortcutPlatform({ platform: "MacIntel" })).toBe("mac");
  });

  it("falls back to the user agent when platform is absent", () => {
    expect(
      detectShortcutPlatform({ userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" }),
    ).toBe("mac");
  });

  it("defaults to windows for Win32 and for an empty source", () => {
    expect(detectShortcutPlatform({ platform: "Win32" })).toBe("windows");
    expect(detectShortcutPlatform({})).toBe("windows");
  });
});

describe("modifierHint", () => {
  it("names each platform's own modifier keys", () => {
    expect(modifierHint("windows")).toContain("Win");
    expect(modifierHint("mac")).toContain("⌘");
    expect(modifierHint("mac")).not.toContain("Ctrl");
  });
});

describe("shouldReapplyShortcut", () => {
  it("re-applies an unchanged accelerator that is not registered", () => {
    expect(shouldReapplyShortcut("Ctrl+Alt+KeyV", "Ctrl+Alt+KeyV", false)).toBe(true);
  });

  it("does nothing when the accelerator is unchanged and still registered", () => {
    expect(shouldReapplyShortcut("Ctrl+Alt+KeyV", "Ctrl+Alt+KeyV", true)).toBe(false);
  });

  it("re-applies whenever the accelerator changed", () => {
    expect(shouldReapplyShortcut("Ctrl+Alt+KeyB", "Ctrl+Alt+KeyV", true)).toBe(true);
    expect(shouldReapplyShortcut("Ctrl+Alt+KeyV", null, false)).toBe(true);
  });
});

describe("shortcutFailureMessage", () => {
  it("names the fallback that survived the refusal", () => {
    const message = shortcutFailureMessage("Ctrl + Alt + B", "HotKey already registered", "Ctrl + Alt + V");

    expect(message).toContain("Ctrl + Alt + B");
    expect(message).toContain("Ctrl + Alt + V is still active");
  });

  it("says push-to-talk is off when nothing survived", () => {
    const message = shortcutFailureMessage("Ctrl + Alt + B", "HotKey already registered", null);

    expect(message).toContain("Ctrl + Alt + B");
    expect(message).toMatch(/off until/);
  });

  it("states the reason the plugin returned rather than a guess", () => {
    expect(shortcutFailureMessage("Ctrl + Alt + B", "RegisterHotKey failed: 1409", null)).toContain(
      "RegisterHotKey failed: 1409",
    );
    expect(shortcutFailureMessage("Ctrl + Alt + B", "HotKey already registered", null)).toContain(
      "HotKey already registered",
    );
    expect(shortcutFailureMessage("Ctrl + Alt + B", "HotKey already registered", null)).not.toContain(
      "another app may already be using it",
    );
  });

  it("falls back to a stated absence when the rejection carried no message", () => {
    expect(shortcutFailureMessage("Ctrl + Alt + B", "   ", null)).toContain("no reason given");
  });
});
