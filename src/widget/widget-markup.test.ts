// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MEETING_DURATION_ID } from "./meeting-indicator";

const DICTATION_DURATION_ID = "widget-duration";

function widgetMarkup(): Document {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  return new DOMParser().parseFromString(html, "text/html");
}

describe("the widget markup the indicator and dictation share", () => {
  it("declares a readout for each of them", () => {
    const document = widgetMarkup();

    expect(document.getElementById(DICTATION_DURATION_ID)).not.toBeNull();
    expect(document.getElementById(MEETING_DURATION_ID)).not.toBeNull();
  });

  it("gives them two different nodes, which is the whole fix", () => {
    const document = widgetMarkup();

    const dictation = document.getElementById(DICTATION_DURATION_ID);
    const meeting = document.getElementById(MEETING_DURATION_ID);

    expect(meeting).not.toBe(dictation);
  });
});
