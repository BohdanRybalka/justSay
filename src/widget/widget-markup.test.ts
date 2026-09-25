// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MEETING_SLOT_CLASS } from "./meeting-indicator";

function widgetMarkup(): Document {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  return new DOMParser().parseFromString(html, "text/html");
}

describe("the widget markup the meeting indicator and the pill share", () => {
  it("keeps the meeting slot inside the root the indicator is handed", () => {
    const document = widgetMarkup();

    const root = document.getElementById("widget")!;

    expect(root.querySelector(`.${MEETING_SLOT_CLASS}`)).not.toBeNull();
  });

  it("keeps the meeting slot out of the slot every pill repaint replaces", () => {
    const document = widgetMarkup();

    const slot = document.querySelector("#widget > .pill-content");

    expect(slot).not.toBeNull();
    expect(slot!.querySelector(`.${MEETING_SLOT_CLASS}`)).toBeNull();
  });
});
