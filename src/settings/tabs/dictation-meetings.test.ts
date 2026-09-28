// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const saveSettingsMock = vi.fn();
vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
}));

const emitSettingsChangedMock = vi.fn();
vi.mock("./dictation", () => ({
  emitSettingsChanged: emitSettingsChangedMock,
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

const { renderDictationMeetings, meetingsHint } = await import("./dictation-meetings");

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: "Ctrl+Alt+KeyV",
    output_dir: "C:/fake",
    stt_mode: "cloud",
    stt_engine: "auto",
    whisper_model_size: "large-v3-turbo",
    whisper_device: "auto",
    ollama_host: "http://localhost:11434",
    cloud_routing_threshold: 30,
    initial_prompt: "",
    gemini_api_key: "",
    groq_api_key: "",
    meeting_consent_acknowledged: false,
    meetings_enabled: false,
    theme: "system",
    display_name: "",
    ...overrides,
  };
}

function render(overrides: Partial<UserSettings> = {}) {
  const container = document.createElement("div");
  const group = renderDictationMeetings(container, buildSettings(overrides));
  const toggle = container.querySelector<HTMLButtonElement>("#meetings-toggle")!;
  const disclosure = container.querySelector<HTMLElement>("#meeting-disclosure")!;
  const consent = container.querySelector<HTMLButtonElement>("#btn-meeting-consent")!;
  const hint = container.querySelector<HTMLElement>("#meetings-hint")!;
  const isOn = () => toggle.getAttribute("aria-checked") === "true";
  return { container, group, toggle, disclosure, consent, hint, isOn };
}

const ACKNOWLEDGED_ON = { meeting_consent_acknowledged: true, meetings_enabled: true };

beforeEach(() => {
  vi.clearAllMocks();
  saveSettingsMock.mockResolvedValue({ settings: buildSettings(), warning: null });
});

describe("renderDictationMeetings — the switch", () => {
  it("is on only when the disclosure is acknowledged and Record meetings is on", () => {
    expect(render(ACKNOWLEDGED_ON).isOn()).toBe(true);
    expect(render({ meeting_consent_acknowledged: true, meetings_enabled: false }).isOn()).toBe(false);
    expect(render({ meeting_consent_acknowledged: false, meetings_enabled: true }).isOn()).toBe(false);
  });

  it("opens the disclosure instead of turning on the first time", () => {
    const view = render();

    expect(view.disclosure.hidden).toBe(true);
    view.toggle.click();

    expect(view.disclosure.hidden).toBe(false);
    expect(view.isOn()).toBe(false);
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("turns on after I understand, saving both answers and telling the widget", async () => {
    const view = render();
    view.toggle.click();
    view.consent.click();

    await vi.waitFor(() => expect(view.isOn()).toBe(true));
    expect(saveSettingsMock).toHaveBeenCalledWith(ACKNOWLEDGED_ON);
    expect(view.disclosure.hidden).toBe(true);
    expect(emitSettingsChangedMock).toHaveBeenCalledTimes(1);
  });

  it("stays off with the disclosure open when I understand could not be saved", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const view = render();
    view.toggle.click();
    view.consent.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(view.isOn()).toBe(false);
    expect(view.disclosure.hidden).toBe(false);
    expect(view.consent.disabled).toBe(false);
    expect(emitSettingsChangedMock).not.toHaveBeenCalled();
  });

  it("turns on straight away once the disclosure was acknowledged", async () => {
    const view = render({ meeting_consent_acknowledged: true, meetings_enabled: false });
    view.toggle.click();

    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledWith({ meetings_enabled: true }));
    expect(view.isOn()).toBe(true);
    expect(view.disclosure.hidden).toBe(true);
  });

  it("turns off and tells the widget, which takes Record a meeting out of the tray", async () => {
    const view = render(ACKNOWLEDGED_ON);
    view.toggle.click();

    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalledTimes(1));
    expect(saveSettingsMock).toHaveBeenCalledWith({ meetings_enabled: false });
    expect(view.isOn()).toBe(false);
    expect(view.toggle.disabled).toBe(false);
  });

  it("goes back on when turning it off could not be saved", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const view = render(ACKNOWLEDGED_ON);
    view.toggle.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(view.isOn()).toBe(true);
  });
});

describe("renderDictationMeetings — where a meeting becomes text", () => {
  it("follows the Cloud / Local choice", () => {
    expect(render({ stt_mode: "cloud" }).hint.textContent).toBe(
      "Right-click the widget and pick Record a meeting · turned into text in the cloud",
    );
    expect(render({ stt_mode: "local" }).hint.textContent).toBe(
      "Right-click the widget and pick Record a meeting · turned into text on this computer",
    );
  });

  it("changes when another mode is picked", () => {
    const view = render({ stt_mode: "cloud" });

    view.group.showMode("local");

    expect(view.hint.textContent).toBe(meetingsHint("local"));
  });
});

describe("renderDictationMeetings — the disclosure (ADR 040 obligation 3)", () => {
  it("states that the user carries the consent obligation", () => {
    const text = render().container.querySelector("#meeting-consent-responsibility")!.textContent!;

    expect(text).toMatch(/responsible/i);
    expect(text).toMatch(/consent/i);
  });

  it("states that Cloud mode sends the other participants' audio to the provider", () => {
    const text = render().container.querySelector("#meeting-consent-cloud")!.textContent!;

    expect(text).toMatch(/cloud/i);
    expect(text).toMatch(/participants/i);
    expect(text).toMatch(/provider/i);
  });
});
