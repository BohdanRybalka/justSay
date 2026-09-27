// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CloudKeyStatus, UserSettings } from "../../api";

const apiMock = {
  updateSettings: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
}));

const saveSettingsMock = vi.fn();
const getCloudKeyStatusMock = vi.fn();

vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
  getCloudKeyStatus: getCloudKeyStatusMock,
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

const { renderKeys } = await import("./keys");

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
    theme: "system",
    ...overrides,
  };
}

const NO_KEYS: CloudKeyStatus = { gemini_key_set: false, groq_key_set: false };

function render(settings: UserSettings, cloud: CloudKeyStatus | null = NO_KEYS): HTMLElement {
  const container = document.createElement("div");
  document.body.replaceChildren(container);
  renderKeys(container, settings, cloud);
  return container;
}

function routeHints(container: HTMLElement): Record<string, string | null> {
  return Object.fromEntries(
    [...container.querySelectorAll<HTMLElement>(".key-row")].map((row) => [
      row.querySelector(".setting-row-title")!.textContent,
      row.querySelector(".route-hint")!.textContent,
    ]),
  );
}

function routeButton(container: HTMLElement, label: string): HTMLButtonElement {
  return [...container.querySelectorAll<HTMLButtonElement>(".route-choice button")].find(
    (button) => button.textContent === label,
  )!;
}

function pressedRoute(container: HTMLElement): string | null | undefined {
  return container.querySelector('.route-choice [aria-pressed="true"]')?.textContent;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("renderKeys — env-sourced key indicator (Bug 1)", () => {
  it("a key present only via cloud-status (.env) renders as key-present, not as unset", () => {
    const container = render(buildSettings(), { gemini_key_set: true, groq_key_set: false });

    expect(container.querySelector("#gemini-status")!.textContent).toContain("environment");
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.readOnly).toBe(true);
    expect(container.querySelector("#gemini-replace")).not.toBeNull();
    expect(container.querySelector("#gemini-save")).toBeNull();
  });

  it("a key stored in settings.json renders masked with Replace and no status line", () => {
    const container = render(buildSettings({ gemini_api_key: "***" }), {
      gemini_key_set: true,
      groq_key_set: false,
    });

    expect(container.querySelector("#gemini-status")!.textContent).toBe("");
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.readOnly).toBe(true);
    expect(input.value).toBe("••••••••••••");
    expect(container.querySelector("#gemini-replace")).not.toBeNull();
  });

  it("no key anywhere renders an empty field with Save and no status line", () => {
    const container = render(buildSettings());

    expect(container.querySelector("#gemini-status")!.textContent).toBe("");
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.readOnly).toBe(false);
    expect(input.value).toBe("");
    expect(input.type).toBe("password");
    expect(container.querySelector("#gemini-save")).not.toBeNull();
    expect(container.querySelector("#gemini-replace")).toBeNull();
  });

  it("a null cloud status (first-load fetch rejected) renders a hedged 'unknown' hint", () => {
    const container = render(buildSettings(), null);

    expect(container.querySelector("#gemini-status")!.textContent).toContain(
      "Cannot verify key status",
    );
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.type).toBe("password");
    expect(container.querySelector("#gemini-save")).not.toBeNull();
    expect(container.querySelector("#gemini-replace")).toBeNull();
  });
});

describe("renderKeys — Replace and Cancel", () => {
  it("Replace turns the row into a focused input with Cancel and Save; Cancel puts the masked key back", () => {
    const container = render(buildSettings({ groq_api_key: "***" }));

    container.querySelector<HTMLButtonElement>("#groq-replace")!.click();

    const input = container.querySelector<HTMLInputElement>("#groq-key-input")!;
    expect(input.readOnly).toBe(false);
    expect(document.activeElement).toBe(input);
    expect(container.querySelector("#groq-cancel")).not.toBeNull();
    expect(container.querySelector<HTMLButtonElement>("#groq-save")!.disabled).toBe(true);

    container.querySelector<HTMLButtonElement>("#groq-cancel")!.click();

    expect(container.querySelector<HTMLInputElement>("#groq-key-input")!.readOnly).toBe(true);
    expect(container.querySelector("#groq-replace")).not.toBeNull();
  });

  it("editing one key leaves the other row as it was", () => {
    const container = render(buildSettings({ groq_api_key: "***", gemini_api_key: "***" }));

    container.querySelector<HTMLButtonElement>("#groq-replace")!.click();

    expect(container.querySelector("#gemini-replace")).not.toBeNull();
  });
});

describe("renderKeys — Save routes through saveSettings (Bug 2)", () => {
  it("the Save handler calls saveSettings from ../settings, not api.updateSettings directly", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ gemini_api_key: "***" }),
      warning: null,
    });
    getCloudKeyStatusMock.mockReturnValue(NO_KEYS);
    const container = render(buildSettings());

    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    input.value = "AIza-new-key";
    input.dispatchEvent(new Event("input"));
    container.querySelector<HTMLButtonElement>("#gemini-save")!.click();

    await vi.waitFor(() => {
      expect(saveSettingsMock).toHaveBeenCalledWith({ gemini_api_key: "AIza-new-key" });
    });
    expect(apiMock.updateSettings).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(container.querySelector("#gemini-replace")).not.toBeNull());
  });

  it("a failed save keeps what was typed and says why", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const container = render(buildSettings());

    const input = container.querySelector<HTMLInputElement>("#groq-key-input")!;
    input.value = "gsk-new";
    input.dispatchEvent(new Event("input"));
    container.querySelector<HTMLButtonElement>("#groq-save")!.click();

    await vi.waitFor(() => {
      expect(container.querySelector("#groq-status")!.textContent).toBe("Error: backend down");
    });
    expect(input.value).toBe("gsk-new");
    expect(input.disabled).toBe(false);
    expect(container.querySelector<HTMLButtonElement>("#groq-save")!.disabled).toBe(false);
  });
});

describe("Recordings go to", () => {
  it("offers Automatic, Groq and Google with the stored choice pressed", () => {
    const container = render(buildSettings({ stt_engine: "groq" }));

    const labels = [...container.querySelectorAll(".route-choice button")].map((b) => b.textContent);
    expect(labels).toEqual(["Automatic", "Groq", "Google"]);
    expect(pressedRoute(container)).toBe("Groq");
  });

  it.each([
    ["auto", { Groq: "Used for short recordings", Google: "Used for long recordings" }],
    ["groq", { Groq: "Used for all recordings", Google: "Used for files Groq can't read" }],
    ["gemini", { Groq: "Not used for recordings", Google: "Used for all recordings" }],
  ] as const)("with %s stored, each key's subtitle says what it is used for", (engine, hints) => {
    const container = render(buildSettings({ stt_engine: engine }));

    expect(routeHints(container)).toEqual(hints);
  });

  it("saves a new choice and the subtitles follow it", async () => {
    saveSettingsMock.mockResolvedValue({ settings: buildSettings({ stt_engine: "gemini" }), warning: null });
    const container = render(buildSettings({ stt_engine: "auto" }));

    routeButton(container, "Google").click();

    expect(saveSettingsMock).toHaveBeenCalledWith({ stt_engine: "gemini" });
    expect(routeHints(container)).toEqual({
      Groq: "Not used for recordings",
      Google: "Used for all recordings",
    });
    expect(pressedRoute(container)).toBe("Google");
  });

  it("goes back to the stored choice and says so when saving fails", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const container = render(buildSettings({ stt_engine: "groq" }));

    routeButton(container, "Automatic").click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalled());
    expect(notifyErrorMock.mock.calls[0][0]).toContain("backend down");
    expect(pressedRoute(container)).toBe("Groq");
    expect(routeHints(container).Groq).toBe("Used for all recordings");
  });

  it("a key row redrawn after a routing change keeps the new subtitle", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ stt_engine: "groq", groq_api_key: "***" }),
      warning: null,
    });
    const container = render(buildSettings({ stt_engine: "auto", groq_api_key: "***" }));

    routeButton(container, "Groq").click();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());
    await Promise.resolve();
    container.querySelector<HTMLButtonElement>("#groq-replace")!.click();

    expect(routeHints(container).Groq).toBe("Used for all recordings");
  });

  it("in Local mode each key says it is not used, whatever the routing choice", () => {
    const container = render(buildSettings({ stt_mode: "local", stt_engine: "groq" }));

    expect(routeHints(container)).toEqual({
      Groq: "Not used while Local is on",
      Google: "Not used while Local is on",
    });

    routeButton(container, "Automatic").click();

    expect(routeHints(container).Groq).toBe("Not used while Local is on");
  });

  it("a failed save overtaken by a newer choice leaves the newer choice on screen", async () => {
    let failFirst!: (e: Error) => void;
    saveSettingsMock
      .mockImplementationOnce(() => new Promise((_, reject) => (failFirst = reject)))
      .mockResolvedValueOnce({ settings: buildSettings({ stt_engine: "gemini" }), warning: null });
    const container = render(buildSettings({ stt_engine: "groq" }));

    routeButton(container, "Automatic").click();
    routeButton(container, "Google").click();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledTimes(2));
    failFirst(new Error("backend down"));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(pressedRoute(container)).toBe("Google");
    expect(routeHints(container).Google).toBe("Used for all recordings");
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });
});
