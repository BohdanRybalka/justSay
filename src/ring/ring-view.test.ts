// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ActionLabel, RingAction } from "./ring-actions";
import {
  BUTTON_HOVER_CLASS,
  buttonAt,
  closeRing,
  followPointer,
  LABEL_SHOWN_CLASS,
  mountRing,
  openRing,
  renderLabels,
  RING_BUTTONS,
  RING_OPEN_CLASS,
} from "./ring-view";

const LABELS: Record<RingAction, ActionLabel> = {
  meeting: { text: "Record a meeting", shortcut: null },
  file: { text: "Transcribe a file", shortcut: null },
  dictate: { text: "Dictate", shortcut: "Ctrl + Alt + V" },
  language: { text: "Language · English", shortcut: null },
  settings: { text: "Settings", shortcut: null },
};

const BUTTON_CENTRE_X = -19;

function rowCentre(index: number): { x: number; y: number } {
  return { x: BUTTON_CENTRE_X, y: (index - 2) * 30 };
}

function ringRoot(): HTMLElement {
  const html = readFileSync(resolve(__dirname, "../../ring.html"), "utf-8");
  document.body.innerHTML = html
    .slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"))
    .replace(/<script[\s\S]*?<\/script>/g, "");
  return document.getElementById("ring")!;
}

function buttons(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(".ring-button")];
}

function hoveredButtons(root: HTMLElement): number[] {
  return buttons(root).flatMap((button, index) => (button.classList.contains(BUTTON_HOVER_CLASS) ? [index] : []));
}

function label(root: HTMLElement): { text: string | null; shown: boolean; row: string } {
  const element = root.querySelector<HTMLElement>(".ring-label")!;
  return {
    text: element.textContent,
    shown: element.classList.contains(LABEL_SHOWN_CLASS),
    row: element.style.getPropertyValue("--row"),
  };
}

describe("the column's geometry", () => {
  it("stacks five buttons with the mic in the middle, where the pill sits", () => {
    expect(RING_BUTTONS.map((button) => button.action)).toEqual([
      "meeting",
      "file",
      "dictate",
      "language",
      "settings",
    ]);
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: 0 })).toBe(2);
  });

  it("finds the button on each row, 30px apart from the top one to the bottom one", () => {
    for (let index = 0; index < 5; index++) {
      expect(buttonAt(rowCentre(index))).toBe(index);
    }
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: -14 })).toBe(2);
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: -16 })).toBe(1);
  });

  it("keeps a button hovered across the column's whole width and its rounded ends", () => {
    expect(buttonAt({ x: -37, y: 0 })).toBe(2);
    expect(buttonAt({ x: 0, y: 0 })).toBe(2);
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: -77 })).toBe(0);
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: 77 })).toBe(4);
  });

  it("hovers nothing beside, above or below the column", () => {
    expect(buttonAt({ x: -38, y: 0 })).toBeNull();
    expect(buttonAt({ x: 1, y: 0 })).toBeNull();
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: -78 })).toBeNull();
    expect(buttonAt({ x: BUTTON_CENTRE_X, y: 78 })).toBeNull();
  });
});

describe("the column on the page", () => {
  let root: HTMLElement;
  const onPick = vi.fn();
  const onDismiss = vi.fn();

  beforeEach(() => {
    onPick.mockClear();
    onDismiss.mockClear();
    root = ringRoot();
    mountRing(root, onPick, onDismiss);
    renderLabels(root, (action) => LABELS[action]);
  });

  it("draws one labelled button per action, in order, with only the mic marked", () => {
    expect(buttons(root).map((button) => button.getAttribute("aria-label"))).toEqual([
      "Record a meeting",
      "Transcribe a file",
      "Dictate · Ctrl + Alt + V",
      "Language · English",
      "Settings",
    ]);
    expect(buttons(root).map((button) => button.style.getPropertyValue("--row"))).toEqual(["0", "1", "2", "3", "4"]);
    expect(buttons(root).map((button) => button.classList.contains("ring-button--mic"))).toEqual([
      false,
      false,
      true,
      false,
      false,
    ]);
  });

  it("marks the button under the pointer and names it beside that row", () => {
    openRing(root);

    followPointer(root, rowCentre(0));
    expect(hoveredButtons(root)).toEqual([0]);
    expect(label(root)).toEqual({ text: "Record a meeting", shown: true, row: "0" });

    followPointer(root, rowCentre(4));
    expect(hoveredButtons(root)).toEqual([4]);
    expect(label(root)).toEqual({ text: "Settings", shown: true, row: "4" });

    followPointer(root, { x: -60, y: 0 });
    expect(hoveredButtons(root)).toEqual([]);
    expect(label(root).shown).toBe(false);
  });

  it("sets the mic's shortcut apart in its label", () => {
    openRing(root);

    followPointer(root, rowCentre(2));

    expect(label(root).text).toBe("Dictate · Ctrl + Alt + V");
    expect(root.querySelector(".ring-label .ring-label-shortcut")?.textContent).toBe(" · Ctrl + Alt + V");
  });

  it("hands the picked button to its caller", () => {
    openRing(root);

    buttons(root)[3].click();

    expect(onPick).toHaveBeenCalledWith("language");
    expect(onDismiss).not.toHaveBeenCalled();
  });

  it("closes on a right click on the mic, and only there", () => {
    openRing(root);

    buttons(root)[0].dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2 }));
    expect(onDismiss).not.toHaveBeenCalled();

    buttons(root)[2].dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2 }));
    expect(onDismiss).toHaveBeenCalledOnce();
    expect(onPick).not.toHaveBeenCalled();
  });

  it("renames the hovered button's label when its state changes", () => {
    openRing(root);
    followPointer(root, rowCentre(3));

    renderLabels(root, (action) =>
      action === "language" ? { text: "Language · Ukrainian", shortcut: null } : LABELS[action],
    );

    expect(buttons(root)[3].getAttribute("aria-label")).toBe("Language · Ukrainian");
    expect(label(root)).toEqual({ text: "Language · Ukrainian", shown: true, row: "3" });
  });

  it("closes back to rest, so the next opening slides in with nothing hovered", () => {
    openRing(root);
    followPointer(root, rowCentre(0));

    closeRing(root);

    expect(root.classList.contains(RING_OPEN_CLASS)).toBe(false);
    expect(root.getAttribute("aria-hidden")).toBe("true");
    expect(hoveredButtons(root)).toEqual([]);
    expect(label(root).shown).toBe(false);

    openRing(root);
    expect(root.classList.contains(RING_OPEN_CLASS)).toBe(true);
    expect(root.getAttribute("aria-hidden")).toBe("false");
    expect(hoveredButtons(root)).toEqual([]);
  });

  it("starts an opening from rest even when the column was never told it closed", () => {
    openRing(root);
    followPointer(root, rowCentre(0));

    openRing(root);

    expect(hoveredButtons(root)).toEqual([]);
  });
});
