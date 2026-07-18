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

// Imported after the mocks above so keys.ts picks up the mocked modules.
const { renderKeys } = await import("./keys");

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: "Ctrl+Alt+KeyV",
    output_dir: "C:/fake",
    stt_mode: "cloud",
    llm_mode: "cloud",
    stt_engine: "auto",
    whisper_model_size: "large-v3-turbo",
    whisper_device: "auto",
    ollama_host: "http://localhost:11434",
    ollama_model: "qwen3:1.7b",
    cloud_routing_threshold: 30,
    initial_prompt: "",
    gemini_api_key: "",
    groq_api_key: "",
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("renderKeys — env-sourced key indicator (Bug 1)", () => {
  it("a key present only via cloud-status (.env) renders as key-present, not the unset hint", () => {
    const container = document.createElement("div");
    const settings = buildSettings({ gemini_api_key: "", groq_api_key: "" });
    const cloud: CloudKeyStatus = { gemini_key_set: true, groq_key_set: false };

    renderKeys(container, settings, cloud);

    const hint = container.querySelector("#gemini-status")!.textContent ?? "";
    expect(hint).not.toContain("No key set");
    expect(hint.toLowerCase()).not.toBe("key stored.");
    expect(hint.toLowerCase()).toContain("environment");

    // The row still renders as masked/disabled with a Replace affordance,
    // like the stored case — only the hint text differs.
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.disabled).toBe(true);
    expect(container.querySelector("#gemini-replace")).not.toBeNull();
  });

  it("a key stored in settings.json still renders as stored, unaffected by cloud status", () => {
    const container = document.createElement("div");
    const settings = buildSettings({ gemini_api_key: "***" });
    const cloud: CloudKeyStatus = { gemini_key_set: true, groq_key_set: false };

    renderKeys(container, settings, cloud);

    expect(container.querySelector("#gemini-status")!.textContent).toBe("Key stored.");
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.disabled).toBe(true);
    expect(container.querySelector("#gemini-replace")).not.toBeNull();
  });

  it("no key in settings.json nor cloud status still renders the original unset hint", () => {
    const container = document.createElement("div");
    const settings = buildSettings({ gemini_api_key: "" });
    const cloud: CloudKeyStatus = { gemini_key_set: false, groq_key_set: false };

    renderKeys(container, settings, cloud);

    expect(container.querySelector("#gemini-status")!.textContent).toBe(
      "No key set — cloud STT will fail.",
    );
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.disabled).toBe(false);
  });

  it("a null cloud status (first-load fetch rejected) renders a hedged 'unknown' hint, not the categorical unset hint", () => {
    // cloud === null means "we don't know" (the cloud-status fetch failed),
    // not "there is no key" -- collapsing the two would render "No key set"
    // from pure ignorance, which is exactly the false-negative class this
    // spec exists to remove (Stage 5 review fix).
    const container = document.createElement("div");
    const settings = buildSettings({ gemini_api_key: "" });

    expect(() => renderKeys(container, settings, null)).not.toThrow();

    const hint = container.querySelector("#gemini-status")!.textContent ?? "";
    expect(hint).not.toBe("No key set — cloud STT will fail.");
    expect(hint).not.toContain("No key set");
    expect(hint).toContain("Cannot verify key status");

    // Must still offer the same input/Save affordance as "unset" -- the
    // user has to be able to enter a key even when status is unverifiable.
    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    expect(input.disabled).toBe(false);
    expect(input.type).toBe("password");
    expect(container.querySelector("#gemini-save")).not.toBeNull();
    expect(container.querySelector("#gemini-replace")).toBeNull();
  });
});

describe("renderKeys — Save routes through saveSettings (Bug 2)", () => {
  it("the Save handler calls saveSettings from ../settings, not api.updateSettings directly", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ gemini_api_key: "***" }),
      warning: null,
    });
    getCloudKeyStatusMock.mockReturnValue({ gemini_key_set: false, groq_key_set: false });

    const container = document.createElement("div");
    const settings = buildSettings({ gemini_api_key: "" });
    const cloud: CloudKeyStatus = { gemini_key_set: false, groq_key_set: false };

    renderKeys(container, settings, cloud);

    const input = container.querySelector<HTMLInputElement>("#gemini-key-input")!;
    input.value = "AIza-new-key";
    input.dispatchEvent(new Event("input"));

    const saveBtn = container.querySelector<HTMLButtonElement>("#gemini-save")!;
    saveBtn.click();

    await vi.waitFor(() => {
      expect(saveSettingsMock).toHaveBeenCalledWith({ gemini_api_key: "AIza-new-key" });
    });
    expect(apiMock.updateSettings).not.toHaveBeenCalled();
  });
});
