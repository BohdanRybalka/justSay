// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";
import { DICTATION_LANGUAGES } from "../../languages";

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

const getMeetingStatusMock = vi.fn();
vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: { ...actual.api, getMeetingStatus: getMeetingStatusMock },
    meetingLevelStream: vi.fn(() => new AbortController()),
  };
});

const { renderMeetings } = await import("./meetings");
const { meetingCardHint } = await import("./meeting-card");

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
    theme: "system",
    display_name: "",
    paste_at_cursor: true,
    previous_language: "",
    ...overrides,
  };
}

function render(overrides: Partial<UserSettings> = {}, windowHidden = true) {
  const container = document.createElement("div");
  const page = renderMeetings(container, buildSettings(overrides), windowHidden);
  const hint = container.querySelector<HTMLElement>("#meeting-hint")!;
  return { container, page, hint };
}

beforeEach(() => {
  vi.clearAllMocks();
  getMeetingStatusMock.mockResolvedValue({ is_recording: false, duration_seconds: 0 });
  saveSettingsMock.mockResolvedValue({ settings: buildSettings(), warning: null });
  getCloudKeyStatusMock.mockReturnValue({ groq_key_set: true });
});

describe("renderMeetings — the page", () => {
  function groupLabels(container: HTMLElement): string[] {
    return [...container.querySelectorAll(".group-label")].map((label) => label.textContent);
  }

  it("is a page of its own, titled Meetings, with the meeting card first", () => {
    const { container } = render();

    expect(container.querySelector(".panel-title")!.textContent).toBe("Meetings");
    expect(container.querySelector(".panel-subtitle")!.textContent).toBe(
      "Record a call in any app and get its text.",
    );
    expect(container.querySelector(":scope > .meeting-card #meeting-start")).not.toBeNull();
    expect(container.querySelector(".meeting-card")!.nextElementSibling!.classList.contains("group-label")).toBe(true);
  });

  it("has no Record meetings switch", () => {
    const { container } = render({ meeting_consent_acknowledged: true });

    expect(container.querySelector("#meetings-toggle")).toBeNull();
    expect(container.textContent).not.toContain("Record meetings");
  });

  it("lets the card read the meeting only while the window is shown", async () => {
    const { page } = render({}, true);
    expect(getMeetingStatusMock).not.toHaveBeenCalled();

    page.resumeResources!();
    await vi.waitFor(() => expect(getMeetingStatusMock).toHaveBeenCalledTimes(1));
    page.releaseResources!();
    page.destroy();
  });

  it("groups the rest as LANGUAGE and MODEL, in that order", () => {
    const { container } = render();

    expect(groupLabels(container)).toEqual(["LANGUAGE", "MODEL"]);
    const [language, model] = container.querySelectorAll(".group-label + .card");
    expect(language.querySelector("#meetings-detect")).not.toBeNull();
    expect(language.querySelector("#meetings-language")).not.toBeNull();
    expect(model.querySelector("#meetings-cloud")).not.toBeNull();
    expect(model.querySelector("#meetings-local")).not.toBeNull();
  });

  it("holds nothing of the Dictation page", () => {
    const { container } = render();

    expect(container.querySelector("#lang-select")).toBeNull();
    expect(container.querySelector("#btn-shortcut")).toBeNull();
    expect(container.querySelector("#dictionary-input")).toBeNull();
  });
});

describe("renderMeetings — where a meeting becomes text", () => {
  function picked(container: HTMLElement): string[] {
    return [...container.querySelectorAll<HTMLButtonElement>('[aria-label="Meeting model"] .mode-row')]
      .filter((row) => row.getAttribute("aria-checked") === "true")
      .map((row) => row.id);
  }

  function rowHint(container: HTMLElement, id: string): HTMLElement {
    return container.querySelector<HTMLElement>(`#${id} .mode-row-hint`)!;
  }

  it("names the meetings choice, not the dictation mode", () => {
    const view = render({ stt_mode: "cloud", meetings_engine: "local" });

    expect(view.hint.textContent).toBe(
      "Your microphone and what this computer plays · turned into text on this computer",
    );
    expect(picked(view.container)).toEqual(["meetings-local"]);
    expect(rowHint(view.container, "meetings-local").textContent).toBe("Stays on this computer");
    expect(rowHint(view.container, "meetings-cloud").textContent).toBe("Everyone's voices go to your cloud provider");
    expect(render({ stt_mode: "local", meetings_engine: "cloud" }).hint.textContent).toBe(meetingCardHint("cloud"));
  });

  it("asks for the Groq key on the Cloud row when there is none", () => {
    getCloudKeyStatusMock.mockReturnValue({ groq_key_set: false });
    const hint = rowHint(render().container, "meetings-cloud");

    expect(hint.textContent).toBe("Add a Groq key in Settings");
    expect(hint.classList.contains("mode-row-alert")).toBe(true);
  });

  it("saves a new choice and names it", async () => {
    const view = render({ meetings_engine: "local" });

    view.container.querySelector<HTMLButtonElement>("#meetings-cloud")!.click();

    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledWith({ meetings_engine: "cloud" }));
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalled());
    expect(view.hint.textContent).toBe(meetingCardHint("cloud"));
    expect(picked(view.container)).toEqual(["meetings-cloud"]);
  });

  it("goes back to the saved choice when the new one could not be saved", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const view = render({ meetings_engine: "local" });

    view.container.querySelector<HTMLButtonElement>("#meetings-cloud")!.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(view.hint.textContent).toBe(meetingCardHint("local"));
    expect(picked(view.container)).toEqual(["meetings-local"]);
  });

  it("takes no second choice while the first is still saving", async () => {
    let finish: (value: unknown) => void = () => {};
    saveSettingsMock.mockReturnValue(new Promise((resolve) => (finish = resolve)));
    const view = render({ meetings_engine: "local" });
    const cloud = view.container.querySelector<HTMLButtonElement>("#meetings-cloud")!;

    cloud.click();
    expect(cloud.getAttribute("aria-disabled")).toBe("true");
    cloud.click();
    finish({ settings: buildSettings(), warning: null });

    await vi.waitFor(() => expect(cloud.hasAttribute("aria-disabled")).toBe(false));
    expect(saveSettingsMock).toHaveBeenCalledTimes(1);
    expect(picked(view.container)).toEqual(["meetings-cloud"]);
  });

  it("saves nothing when the chosen row is picked again", () => {
    render({ meetings_engine: "local" }).container.querySelector<HTMLButtonElement>("#meetings-local")!.click();

    expect(saveSettingsMock).not.toHaveBeenCalled();
  });
});

describe("renderMeetings — the language a meeting is heard in", () => {
  function languageView(overrides: Partial<UserSettings>) {
    const { container } = render(overrides);
    const detect = container.querySelector<HTMLButtonElement>("#meetings-detect")!;
    const row = container.querySelector<HTMLElement>("#meetings-language-row")!;
    const select = container.querySelector<HTMLSelectElement>("#meetings-language")!;
    const detecting = () => detect.getAttribute("aria-checked") === "true";
    return { detect, row, select, detecting };
  }

  function choose(select: HTMLSelectElement, value: string) {
    select.value = value;
    select.dispatchEvent(new Event("change"));
  }

  it("lists only languages, showing the saved one with detection off", () => {
    const view = languageView({ meetings_language: "pl" });

    expect([...view.select.options].map((o) => o.textContent)).toEqual(DICTATION_LANGUAGES.map((l) => l.label));
    expect(view.select.value).toBe("pl");
    expect(view.detecting()).toBe(false);
    expect(view.row.hidden).toBe(false);
  });

  it("hides the list while detecting, keeping the dictation language ready in it", () => {
    const view = languageView({ meetings_language: "auto", language: "en" });

    expect(view.detecting()).toBe(true);
    expect(view.row.hidden).toBe(true);
    expect(view.select.value).toBe("en");
  });

  it("turns detection on and off, coming back to the language picked before", async () => {
    const view = languageView({ meetings_language: "pl" });

    view.detect.click();
    expect(view.row.hidden).toBe(true);
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledWith({ meetings_language: "auto" }));
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalledTimes(1));
    view.detect.click();

    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenLastCalledWith({ meetings_language: "pl" }));
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalledTimes(2));
    expect(view.detecting()).toBe(false);
    expect(view.row.hidden).toBe(false);
  });

  it("saves a new language and goes back to it, not the first one, after a failed save", async () => {
    const view = languageView({ meetings_language: "uk" });

    choose(view.select, "en");
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalled());
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    choose(view.select, "de");

    expect(saveSettingsMock).toHaveBeenCalledWith({ meetings_language: "en" });
    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(view.select.value).toBe("en");
  });

  it("stays on the saved language when detection could not be turned on", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const view = languageView({ meetings_language: "uk" });

    view.detect.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(view.detecting()).toBe(false);
    expect(view.row.hidden).toBe(false);
    expect(view.select.value).toBe("uk");
  });
});
