import { describe, expect, it } from "vitest";
import { dictationResultView } from "./done-status";

describe("dictationResultView", () => {
  it("counts the words the way the backend's word_count does", () => {
    expect(dictationResultView({ text: "  hello   world\nagain " }, "copied")).toEqual({
      kind: "done",
      words: 3,
      delivery: "copied",
    });
  });

  it("says pasted only when the text was pasted", () => {
    expect(dictationResultView({ text: "hello world" }, "pasted")).toEqual({
      kind: "done",
      words: 2,
      delivery: "pasted",
    });
  });

  it("raises a one-off alert when the text reached neither the app nor the clipboard", () => {
    expect(dictationResultView({ text: "hello world" }, "failed")).toEqual({
      kind: "alert",
      label: "Copy failed",
    });
  });

  it("shows nothing for empty or whitespace-only text", () => {
    expect(dictationResultView({ text: "   " }, "failed")).toBeNull();
    expect(dictationResultView({ text: "", discarded_reason: null }, "failed")).toBeNull();
  });

  it("reads a silence discard as No speech, although its text is empty too", () => {
    expect(dictationResultView({ text: "", discarded_reason: "silence" }, "failed")).toEqual({
      kind: "noSpeech",
    });
  });
});
