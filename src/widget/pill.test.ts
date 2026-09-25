// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MEETING_DURATION_ID } from "./meeting-indicator";
import { PILL_HOVER_CLASS, renderPill, type PillView } from "./pill";

const STATE_MODIFIERS = ["pill--rest", "pill--live", "pill--alert"];

function pillFromWidgetMarkup(): HTMLElement {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  document.body.innerHTML = html
    .slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"))
    .replace(/<script[\s\S]*?<\/script>/g, "");
  return document.getElementById("widget")!;
}

function shownLabels(pill: HTMLElement): (string | null)[] {
  return [...pill.querySelectorAll(".pill-content .pill-label")].map((label) => label.textContent);
}

function modifiers(pill: HTMLElement): string[] {
  return STATE_MODIFIERS.filter((modifier) => pill.classList.contains(modifier));
}

const EVERY_VIEW: PillView[] = [
  { kind: "rest", hint: "Ctrl + Alt + V" },
  { kind: "listening", label: "Recording", readout: "1.2s" },
  { kind: "working", label: "Processing", readout: "" },
  { kind: "done", label: "Copied", readout: "3.4s" },
  { kind: "alert", label: "Offline" },
];

describe("the pill at rest", () => {
  it("is the microphone with the shortcut hint beside it", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "rest", hint: "Ctrl + Alt + V" });

    expect(modifiers(pill)).toEqual(["pill--rest"]);
    expect(pill.querySelector(".pill-content use")?.getAttribute("href")).toBe("#mic");
    expect(pill.querySelector(".pill-content .pill-hint")?.textContent).toBe("Ctrl + Alt + V");
  });

  it("writes the hint as text, because a shortcut comes from the user's settings", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "rest", hint: "<b>Ctrl</b>" });

    expect(pill.querySelector(".pill-hint b")).toBeNull();
    expect(pill.querySelector(".pill-hint")?.textContent).toBe("<b>Ctrl</b>");
  });
});

describe("the pill while something happens", () => {
  it.each([
    [{ kind: "listening", label: "Recording", readout: "1.2s" }, ["pill--live"], ["Recording", "1.2s"]],
    [{ kind: "working", label: "Processing", readout: "" }, [], ["Processing"]],
    [{ kind: "done", label: "Copied", readout: "3.4s" }, [], ["Copied", "3.4s"]],
    [{ kind: "alert", label: "Offline" }, ["pill--alert"], ["Offline"]],
  ] as [PillView, string[], string[]][])(
    "shows %o with its own modifier and labels",
    (view, expectedModifiers, expectedLabels) => {
      const pill = pillFromWidgetMarkup();
      renderPill(pill, { kind: "rest", hint: "Ctrl + Alt + V" });

      renderPill(pill, view);

      expect(modifiers(pill)).toEqual(expectedModifiers);
      expect(shownLabels(pill)).toEqual(expectedLabels);
      expect(pill.querySelector(".pill-content .icon")).toBeNull();
    },
  );

  it("sets a readout in the tabular figures every number uses", () => {
    const pill = pillFromWidgetMarkup();

    renderPill(pill, { kind: "listening", label: "Recording", readout: "1.2s" });

    expect(pill.querySelector(".pill-content .pill-readout")?.classList.contains("num")).toBe(true);
  });
});

describe("what a repaint leaves alone", () => {
  it("keeps the hover look, which only the shell's cursor loop sets", () => {
    const pill = pillFromWidgetMarkup();
    pill.classList.add(PILL_HOVER_CLASS);

    for (const view of EVERY_VIEW) {
      renderPill(pill, view);
      expect(pill.classList.contains(PILL_HOVER_CLASS)).toBe(true);
    }
  });

  it("keeps the meeting clock's own readout, so a running meeting never blanks", () => {
    const pill = pillFromWidgetMarkup();
    const meetingClock = document.getElementById(MEETING_DURATION_ID)!;
    meetingClock.textContent = "12:04";

    for (const view of EVERY_VIEW) {
      renderPill(pill, view);
      expect(document.getElementById(MEETING_DURATION_ID)).toBe(meetingClock);
      expect(meetingClock.textContent).toBe("12:04");
    }
  });
});
