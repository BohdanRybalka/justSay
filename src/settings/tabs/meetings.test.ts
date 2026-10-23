// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";
import { DICTATION_LANGUAGES } from "../../languages";

const saveSettingsMock = vi.fn();
vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
}));

const emitSettingsChangedMock = vi.fn();
vi.mock("./dictation", () => ({
  emitSettingsChanged: emitSettingsChangedMock,
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../notify")>();
  return { ...actual, notifyError: notifyErrorMock };
});

const getMeetingStatusMock = vi.fn();
const getHistoryMock = vi.fn();
const jobsMock = vi.fn();
vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getMeetingStatus: getMeetingStatusMock,
      getHistory: getHistoryMock,
      jobs: jobsMock,
    },
    meetingLevelStream: vi.fn(() => new AbortController()),
  };
});

const { buildEntry, FakeResizeObserver, pageOf } = await import("../history-page-stub.test-helper");
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
    meetings_language: "uk",
    files_language: "auto",
    theme: "system",
    display_name: "",
    paste_at_cursor: true,
    previous_language: "",
    ...overrides,
  };
}

const pages: { destroy: () => void }[] = [];

function render(overrides: Partial<UserSettings> = {}, windowHidden = true, openHistory = vi.fn()) {
  const container = document.createElement("div");
  const page = renderMeetings(container, buildSettings(overrides), windowHidden, openHistory);
  pages.push(page);
  const hint = container.querySelector<HTMLElement>("#meeting-hint")!;
  return { container, page, hint };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  getHistoryMock.mockResolvedValue(pageOf([], 0, null));
  jobsMock.mockResolvedValue([]);
  getMeetingStatusMock.mockResolvedValue({ is_recording: false, duration_seconds: 0 });
  saveSettingsMock.mockResolvedValue({ settings: buildSettings(), warning: null });
});

afterEach(() => {
  pages.splice(0).forEach((page) => page.destroy());
  vi.unstubAllGlobals();
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
    expect(container.querySelector(".meeting-card")!.nextElementSibling!.querySelector(".recent")).not.toBeNull();
  });

  it("lists the latest meetings under the card, and sends All in History to the meetings filter", async () => {
    getHistoryMock.mockResolvedValue(pageOf([{ ...buildEntry("m1"), source: "meeting" }], 1, null));
    const openHistory = vi.fn();
    const { container } = render({}, true, openHistory);

    await vi.waitFor(() => expect(container.querySelector(".recent .entry--meeting")).not.toBeNull());
    expect(container.querySelector(".recent .group-label")!.textContent).toBe("RECENT MEETINGS");
    expect(getHistoryMock).toHaveBeenCalledWith(3, null, { source: "meeting", starred: false });

    container.querySelector<HTMLButtonElement>(".see-all")!.click();

    expect(openHistory).toHaveBeenCalledExactlyOnceWith("meeting");
  });

  it("stops reading the latest meetings for good when the page is left", async () => {
    vi.useFakeTimers();
    try {
      const { page } = render({}, false);
      await vi.advanceTimersByTimeAsync(0);
      const reads = getHistoryMock.mock.calls.length;

      page.destroy();
      await vi.advanceTimersByTimeAsync(20000);

      expect(getHistoryMock).toHaveBeenCalledTimes(reads);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops reading the latest meetings while the window is hidden", async () => {
    vi.useFakeTimers();
    try {
      const { page } = render({}, false);
      await vi.advanceTimersByTimeAsync(0);
      const reads = getHistoryMock.mock.calls.length;

      page.releaseResources!();
      await vi.advanceTimersByTimeAsync(20000);
      expect(getHistoryMock).toHaveBeenCalledTimes(reads);

      page.resumeResources!();
      await vi.advanceTimersByTimeAsync(0);
      expect(getHistoryMock).toHaveBeenCalledTimes(reads + 1);
    } finally {
      vi.useRealTimers();
    }
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

  it("groups the rest as LANGUAGE", () => {
    const { container } = render();

    expect(groupLabels(container)).toEqual(["RECENT MEETINGS", "LANGUAGE"]);
    const [language] = container.querySelectorAll(".group-label + .card");
    expect(language.querySelector("#meetings-detect")).not.toBeNull();
    expect(language.querySelector("#meetings-language")).not.toBeNull();
  });

  it("holds nothing of the Dictation page", () => {
    const { container } = render();

    expect(container.querySelector("#lang-select")).toBeNull();
    expect(container.querySelector("#btn-shortcut")).toBeNull();
    expect(container.querySelector("#dictionary-input")).toBeNull();
  });
});

describe("renderMeetings — where a meeting becomes text", () => {
  it("names the dictation mode in the card and on the page, and says where to change it", () => {
    const local = render({ stt_mode: "local" });
    expect(local.hint.textContent).toBe(meetingCardHint("local"));
    expect(local.container.querySelector("#model-note")!.textContent).toBe(
      "Turned into text on this computer. Change this on the Dictation page.",
    );

    const cloud = render({ stt_mode: "cloud" });
    expect(cloud.hint.textContent).toBe(
      "Your microphone and what this computer plays · turned into text in the cloud",
    );
    expect(cloud.container.querySelector("#model-note")!.textContent).toBe(
      "Turned into text in the cloud. Change this on the Dictation page.",
    );
  });

  it("offers no model choice of its own, so nothing here can change where meetings go", () => {
    const { container } = render();

    expect(container.querySelector('[aria-label="Meeting model"]')).toBeNull();
    expect(container.querySelector("#meetings-cloud, #meetings-local")).toBeNull();
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
