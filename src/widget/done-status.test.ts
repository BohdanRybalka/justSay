import { describe, expect, it } from "vitest";
import { dictationResultView } from "./done-status";

describe("dictationResultView", () => {
  it("counts the words the way the backend's word_count does", () => {
    expect(
      dictationResultView({ text: "  hello   world\nagain ", copied_to_clipboard: true }),
    ).toEqual({ kind: "done", words: 3 });
  });

  it("raises a one-off alert when the text could not reach the clipboard", () => {
    expect(dictationResultView({ text: "hello world", copied_to_clipboard: false })).toEqual({
      kind: "alert",
      label: "Copy failed",
    });
  });

  it("shows nothing for empty or whitespace-only text", () => {
    expect(dictationResultView({ text: "   ", copied_to_clipboard: true })).toBeNull();
    expect(
      dictationResultView({ text: "", copied_to_clipboard: true, discarded_reason: null }),
    ).toBeNull();
  });

  it("reads a silence discard as No speech, although its text is empty too", () => {
    expect(
      dictationResultView({ text: "", copied_to_clipboard: false, discarded_reason: "silence" }),
    ).toEqual({ kind: "noSpeech" });
  });
});
