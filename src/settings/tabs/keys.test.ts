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

const { renderKeys } = await import("./keys");

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
    meetings_enabled: false,
    theme: "system",
    display_name: "",
    paste_at_cursor: true,
    previous_language: "",
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

function useHints(container: HTMLElement): Record<string, string | null> {
  return Object.fromEntries(
    [...container.querySelectorAll<HTMLElement>(".key-row")].map((row) => [
      row.querySelector(".setting-row-title")!.textContent,
      row.querySelector(".use-hint")!.textContent,
    ]),
  );
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

describe("What each key is for", () => {
  it("shows only the two key rows, with no choice of where recordings go", () => {
    const container = render(buildSettings());

    const titles = [...container.querySelectorAll(".setting-row-title")].map((t) => t.textContent);
    expect(titles).toEqual(["Groq", "Google"]);
    expect(container.querySelector(".route-choice")).toBeNull();
  });

  it("says Groq turns recordings into text and Google serves History search", () => {
    const container = render(buildSettings());

    expect(useHints(container)).toEqual({
      Groq: "Turns your recordings into text",
      Google: "Used for History search",
    });
  });

  it("in Local mode each key says it is not used", () => {
    const container = render(buildSettings({ stt_mode: "local" }));

    expect(useHints(container)).toEqual({
      Groq: "Not used while Local is on",
      Google: "Not used while Local is on",
    });
  });

  it("a key row redrawn after a save keeps its subtitle", async () => {
    saveSettingsMock.mockResolvedValue({ settings: buildSettings({ groq_api_key: "***" }), warning: null });
    const container = render(buildSettings());

    const input = container.querySelector<HTMLInputElement>("#groq-key-input")!;
    input.value = "gsk-new";
    input.dispatchEvent(new Event("input"));
    container.querySelector<HTMLButtonElement>("#groq-save")!.click();

    await vi.waitFor(() => expect(container.querySelector("#groq-replace")).not.toBeNull());
    expect(useHints(container).Groq).toBe("Turns your recordings into text");
  });
});
