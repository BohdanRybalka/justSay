// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const apiMock = {
  getStorageInfo: vi.fn(),
  audioStop: vi.fn(),
  audioStatus: vi.fn(),
  audioStart: vi.fn(),
  cleanupTemp: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
  levelStream: vi.fn(() => ({ abort: vi.fn() })),
}));

const saveSettingsMock = vi.fn();
const getCloudKeyStatusMock = vi.fn();

vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
  getCloudKeyStatus: getCloudKeyStatusMock,
}));

vi.mock("./keys", () => ({
  renderKeys: vi.fn(() => () => {}),
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

const emitMock = vi.fn();
vi.mock("@tauri-apps/api/event", () => ({
  emit: emitMock,
}));

const { renderGeneral } = await import("./general");

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
  apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
});

describe("renderGeneral — Dictation Language change (Bug 3)", () => {
  it("changing the language select persists the value AND emits settings-changed", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ language: "en" }),
      warning: null,
    });

    const container = document.createElement("div");
    renderGeneral(container, buildSettings({ language: "uk" }));

    const langSelect = container.querySelector<HTMLSelectElement>("#lang-select")!;
    langSelect.value = "en";
    langSelect.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(saveSettingsMock).toHaveBeenCalledWith({ language: "en" });
    });
    await vi.waitFor(() => {
      expect(emitMock).toHaveBeenCalledWith("settings-changed");
    });
  });

  it("a failed language save is reported via notifyError rather than swallowed", async () => {
    saveSettingsMock.mockRejectedValue(new Error("network down"));

    const container = document.createElement("div");
    renderGeneral(container, buildSettings({ language: "uk" }));

    const langSelect = container.querySelector<HTMLSelectElement>("#lang-select")!;
    langSelect.value = "en";
    langSelect.dispatchEvent(new Event("change"));

    await vi.waitFor(() => {
      expect(notifyErrorMock).toHaveBeenCalled();
    });
    expect(emitMock).not.toHaveBeenCalled();
  });
});

describe("renderGeneral — history path is separated from temp cleanup (spec 054)", () => {
  function groupOf(container: HTMLElement, selector: string): HTMLElement {
    return container.querySelector<HTMLElement>(selector)!.closest(".setting-group")!;
  }

  it("the history path and the Clear Temp Files button live in different groups", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    const pathGroup = groupOf(container, "#output-dir");
    const cleanupGroup = groupOf(container, "#btn-cleanup");

    expect(pathGroup).not.toBe(cleanupGroup);
  });

  it("each group carries its own label so neither reads as the other's directory", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    const pathLabel = groupOf(container, "#output-dir").querySelector(".setting-label")!;
    const cleanupLabel = groupOf(container, "#btn-cleanup").querySelector(".setting-label")!;

    expect(pathLabel.textContent).not.toBe(cleanupLabel.textContent);
    expect(pathLabel.textContent).toMatch(/history/i);
    expect(cleanupLabel.textContent).toMatch(/audio/i);
  });
});
