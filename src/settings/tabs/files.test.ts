// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { DROP_AREA_CLASS, DROP_HINT } from "../file-transcription";
import { renderFiles } from "./files";

function render(pick: () => void = () => {}) {
  const container = document.createElement("div");
  renderFiles(container, pick);
  return container;
}

describe("renderFiles — the page", () => {
  it("is a page of its own, titled Files, with the new subtitle", () => {
    const container = render();

    expect(container.querySelector(".panel-title")!.textContent).toBe("Files");
    expect(container.querySelector(".panel-subtitle")!.textContent).toBe("Turn a recording into text.");
  });

  it("holds one drop area that says what to drop and which files fit", () => {
    const container = render();

    const areas = container.querySelectorAll(`.${DROP_AREA_CLASS}`);
    expect(areas).toHaveLength(1);
    expect(areas[0].querySelector(".drop-area-title")!.textContent).toBe("Drop an audio or video file here");
    expect(areas[0].querySelector(".drop-area-hint")!.textContent).toBe(DROP_HINT);
  });

  it("opens the file dialog when Choose a file is pressed, once per press", () => {
    const pick = vi.fn();
    const container = render(pick);
    const choose = container.querySelector<HTMLButtonElement>(`.${DROP_AREA_CLASS} button`)!;

    expect(choose.textContent).toBe("Choose a file");
    expect(pick).not.toHaveBeenCalled();

    choose.click();
    choose.click();

    expect(pick).toHaveBeenCalledTimes(2);
  });
});
