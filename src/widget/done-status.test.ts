import { describe, expect, it } from "vitest";
import { dictationResultView } from "./done-status";

describe("dictationResultView", () => {
  it("counts the words the way the backend's word_count does", () => {
    expect(dictationResultView({ text: "  hello   world\nagain " }, true)).toEqual({
      kind: "done",
      words: 3,
    });
  });

  it("raises a one-off alert when the text could not reach the clipboard", () => {
    expect(dictationResultView({ text: "hello world" }, false)).toEqual({
      kind: "alert",
      label: "Copy failed",
    });
  });

  it("shows nothing for empty or whitespace-only text", () => {
    expect(dictationResultView({ text: "   " }, false)).toBeNull();
    expect(dictationResultView({ text: "", discarded_reason: null }, false)).toBeNull();
  });

  it("reads a silence discard as No speech, although its text is empty too", () => {
    expect(dictationResultView({ text: "", discarded_reason: "silence" }, false)).toEqual({
      kind: "noSpeech",
    });
  });
});
