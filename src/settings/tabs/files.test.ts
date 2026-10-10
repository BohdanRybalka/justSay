// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildEntry, FakeResizeObserver, pageOf } from "../history-page-stub.test-helper";
import type { Recent } from "../recent";

const apiMock = { getHistory: vi.fn(), jobs: vi.fn() };

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

const { DROP_AREA_CLASS, DROP_HINT } = await import("../file-transcription");
const { renderFiles } = await import("./files");

const mounted: Recent[] = [];

function render(pick: () => void = () => {}, openHistory: (source: string) => void = () => {}) {
  const container = document.createElement("div");
  mounted.push(renderFiles(container, pick, openHistory));
  return container;
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  apiMock.getHistory.mockResolvedValue(pageOf([], 0, null));
  apiMock.jobs.mockResolvedValue([]);
});

afterEach(() => {
  mounted.splice(0).forEach((page) => page.destroy());
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

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

  it("lists the latest files under the drop area, and sends All in History to the files filter", async () => {
    apiMock.getHistory.mockResolvedValue(
      pageOf([{ ...buildEntry("f1"), source: "file", source_name: "memo.ogg" }], 1, null),
    );
    const openHistory = vi.fn();
    const container = render(() => {}, openHistory);

    await vi.waitFor(() => expect(container.querySelector(".recent .entry--file")).not.toBeNull());
    expect(container.querySelector(`.${DROP_AREA_CLASS}`)!.nextElementSibling!.classList.contains("recent")).toBe(true);
    expect(container.querySelector(".recent .group-label")!.textContent).toBe("RECENT FILES");
    expect(apiMock.getHistory).toHaveBeenCalledWith(3, null, { source: "file", starred: false });

    container.querySelector<HTMLButtonElement>(".see-all")!.click();

    expect(openHistory).toHaveBeenCalledExactlyOnceWith("file");
  });
});

describe("files.css", () => {
  it("turns the drop area orange under the class the window carries while a file is dragged over it", () => {
    const css = readFileSync(resolve(__dirname, "files.css"), "utf-8");

    const rule = css.match(/\.dragging-file \.drop-area\s*\{([^}]*)\}/);

    expect(rule, "no rule for a drop area while a file is dragged").not.toBeNull();
    expect(rule![1]).toContain("border-color: var(--orange)");
  });
});
