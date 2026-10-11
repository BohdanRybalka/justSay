// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";
import { DICTATION_LANGUAGES } from "../../languages";
import { buildEntry, FakeResizeObserver, pageOf } from "../history-page-stub.test-helper";
import type { Recent } from "../recent";

const apiMock = { getHistory: vi.fn(), jobs: vi.fn() };
const saveSettingsMock = vi.fn();
const getCloudKeyStatusMock = vi.fn();
vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
  getCloudKeyStatus: getCloudKeyStatusMock,
}));

const emitSettingsChangedMock = vi.fn();
vi.mock("./dictation", () => ({
  emitSettingsChanged: emitSettingsChangedMock,
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock };
});

const { DROP_AREA_CLASS, DROP_HINT } = await import("../file-transcription");
const { renderFiles } = await import("./files");

const mounted: Recent[] = [];

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: "Ctrl+Alt+KeyV",
    output_dir: "C:/fake",
    stt_mode: "cloud",
    whisper_model_size: "large-v3-turbo",
    whisper_device: "auto",
    ollama_host: "http://localhost:11434",
    initial_prompt: "",
    gemini_api_key: "",
    groq_api_key: "",
    meeting_consent_acknowledged: false,
    meetings_engine: "local",
    meetings_language: "uk",
    files_engine: "cloud",
    files_language: "auto",
    theme: "system",
    display_name: "",
    paste_at_cursor: true,
    previous_language: "",
    ...overrides,
  };
}

function render(
  pick: () => void = () => {},
  openHistory: (source: string) => void = () => {},
  overrides: Partial<UserSettings> = {},
) {
  const container = document.createElement("div");
  mounted.push(renderFiles(container, buildSettings(overrides), pick, openHistory));
  return container;
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  apiMock.getHistory.mockResolvedValue(pageOf([], 0, null));
  apiMock.jobs.mockResolvedValue([]);
  saveSettingsMock.mockResolvedValue({ settings: buildSettings(), warning: null });
  getCloudKeyStatusMock.mockReturnValue({ groq_key_set: true });
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

describe("renderFiles — the language and the model of a file", () => {
  function picked(container: HTMLElement): string[] {
    return [...container.querySelectorAll<HTMLButtonElement>('[aria-label="File model"] .mode-row')]
      .filter((row) => row.getAttribute("aria-checked") === "true")
      .map((row) => row.id);
  }

  function rowHint(container: HTMLElement, id: string): HTMLElement {
    return container.querySelector<HTMLElement>(`#${id} .mode-row-hint`)!;
  }

  function choose(select: HTMLSelectElement, value: string) {
    select.value = value;
    select.dispatchEvent(new Event("change"));
  }

  it("puts LANGUAGE and MODEL after the latest files", () => {
    const container = render();

    const labels = [...container.querySelectorAll(".group-label")].map((label) => label.textContent);
    expect(labels).toEqual(["RECENT FILES", "LANGUAGE", "MODEL"]);
    const [language, model] = container.querySelectorAll(".group-label + .card");
    expect(language.querySelector("#files-detect")).not.toBeNull();
    expect(language.querySelector("#files-language")).not.toBeNull();
    expect(model.querySelector("#files-cloud")).not.toBeNull();
    expect(model.querySelector("#files-local")).not.toBeNull();
  });

  it("detects the language until one is chosen, saying each file is heard in its own", () => {
    const container = render();

    const detect = container.querySelector<HTMLButtonElement>("#files-detect")!;
    expect(detect.getAttribute("aria-checked")).toBe("true");
    expect(container.querySelector<HTMLElement>("#files-language-row")!.hidden).toBe(true);
    expect(container.textContent).toContain("Each file is heard in its own language");
  });

  it("shows a chosen file language with its list, titled File language", () => {
    const container = render(undefined, undefined, { files_language: "pl", language: "en" });

    const select = container.querySelector<HTMLSelectElement>("#files-language")!;
    expect(container.querySelector("#files-language-row .setting-row-title")!.textContent).toBe("File language");
    expect([...select.options].map((o) => o.textContent)).toEqual(DICTATION_LANGUAGES.map((l) => l.label));
    expect(select.value).toBe("pl");
    expect(container.querySelector<HTMLElement>("#files-language-row")!.hidden).toBe(false);
  });

  it("saves the file language, never the meeting or dictation one", async () => {
    const container = render();

    container.querySelector<HTMLButtonElement>("#files-detect")!.click();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledWith({ files_language: "uk" }));
    choose(container.querySelector<HTMLSelectElement>("#files-language")!, "en");

    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenLastCalledWith({ files_language: "en" }));
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalledTimes(2));
    expect(saveSettingsMock.mock.calls.flatMap(([saved]) => Object.keys(saved))).toEqual([
      "files_language",
      "files_language",
    ]);
  });

  it("names the Cloud and Local rows and ticks the one the files use, not the dictation mode", () => {
    const container = render(undefined, undefined, { stt_mode: "local", files_engine: "cloud" });

    expect(picked(container)).toEqual(["files-cloud"]);
    expect(rowHint(container, "files-cloud").textContent).toBe("Your API keys · fastest and most accurate");
    expect(rowHint(container, "files-local").textContent).toBe("Stays on this computer");
  });

  it("asks for the Groq key on the Cloud row when there is none", () => {
    getCloudKeyStatusMock.mockReturnValue({ groq_key_set: false });
    const hint = rowHint(render(), "files-cloud");

    expect(hint.textContent).toBe("Add a Groq key in Settings");
    expect(hint.classList.contains("mode-row-alert")).toBe(true);
  });

  it("saves the files engine, never the meetings or dictation one", async () => {
    const container = render(undefined, undefined, { files_engine: "cloud" });

    container.querySelector<HTMLButtonElement>("#files-local")!.click();

    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledExactlyOnceWith({ files_engine: "local" }));
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalled());
    expect(picked(container)).toEqual(["files-local"]);
  });

  it("goes back to the saved engine when the new one could not be saved", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const container = render(undefined, undefined, { files_engine: "cloud" });

    container.querySelector<HTMLButtonElement>("#files-local")!.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(picked(container)).toEqual(["files-cloud"]);
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
