// @vitest-environment jsdom
//
// Stage 3 review fix (spec 027, YELLOW finding): saveSettings()'s
// cloud-status refetch, on failure, must RETAIN the previous cached value
// rather than nulling it out -- nulling would mislabel an untouched
// env-sourced row as "No key set" for the rest of the render pass, which is
// exactly the false-negative bug this spec exists to eliminate.
//
// This test drives the REAL settings.ts + REAL keys.ts through an actual
// simulated user interaction (fill input, click Save) rather than mocking
// ../settings the way keys.test.ts/general.test.ts do, because the bug lives
// inside settings.ts's own caching logic -- a test that mocks ../settings
// away cannot exercise the code under test. settings.ts self-invokes init()
// at module import time, so the DOM elements it reads (#tab-content,
// #backend-status) are created before the dynamic import below, and all
// backend calls are mocked so init()'s background work settles
// deterministically before this test drives its own interaction.
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../api";

const apiMock = {
  health: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
  cloudKeyStatus: vi.fn(),
  getStorageInfo: vi.fn(),
  audioStop: vi.fn(),
  audioStatus: vi.fn(),
  audioStart: vi.fn(),
  cleanupTemp: vi.fn(),
};

vi.mock("../api", () => ({
  api: apiMock,
  levelStream: vi.fn(() => ({ abort: vi.fn() })),
}));

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
  // Each test dynamically re-imports "./settings" fresh (below), so the
  // module registry must be reset too -- otherwise a second test would
  // reuse the first test's module instance, whose captured `tabContent`
  // element reference is now detached from the DOM rebuilt here.
  vi.resetModules();
  document.body.innerHTML = `
    <div id="tab-content"></div>
    <div id="backend-status"></div>
  `;
});

describe("saveSettings — cloud-status refetch failure retains, does not null (Stage 3 fix)", () => {
  it("a failed refetch after saving Gemini leaves the untouched env-sourced Groq row rendering as env, not unset", async () => {
    // Baseline: Gemini unset in settings.json, Groq active only via .env
    // (cloud-status). This is what init()'s own loadSettings() call picks
    // up as the standing mock default.
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings({ gemini_api_key: "", groq_api_key: "" }));
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const settingsModule = await import("./settings");

    // Wait for init()'s background loadSettings() + switchTab("general") to
    // settle -- both run synchronously in the same continuation once
    // loadSettings() resolves, so by the time getSettings() is non-null the
    // General tab (incl. the Keys section) has already been rendered.
    await vi.waitFor(() => {
      expect(settingsModule.getSettings()).not.toBeNull();
      expect(document.getElementById("gemini-save")).not.toBeNull();
    });

    // Sanity-check the baseline before touching anything: Groq shows the
    // env hint from the very first paint.
    expect(document.getElementById("groq-status")!.textContent).toContain("environment");
    expect(document.getElementById("gemini-status")!.textContent).toBe(
      "No key set — cloud STT will fail.",
    );

    // Now simulate the user saving a Gemini key, with the cloud-status
    // refetch that follows a key save specifically failing.
    apiMock.updateSettings.mockResolvedValueOnce({
      settings: buildSettings({ gemini_api_key: "***", groq_api_key: "" }),
      warning: null,
    });
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new Error("network down"));

    const geminiInput = document.getElementById("gemini-key-input") as HTMLInputElement;
    geminiInput.value = "AIza-real-key";
    geminiInput.dispatchEvent(new Event("input"));
    (document.getElementById("gemini-save") as HTMLButtonElement).click();

    // Wait for the post-save re-render to land.
    await vi.waitFor(() => {
      expect(document.getElementById("gemini-status")!.textContent).toBe("Key stored.");
    });

    // The core assertion: Groq — untouched by this save, sourced only from
    // the now-failed-to-refetch cloud status — must still show the env
    // hint, not regress to "No key set".
    const groqHint = document.getElementById("groq-status")!.textContent ?? "";
    expect(groqHint).not.toContain("No key set");
    expect(groqHint).toContain("environment");

    // And the cache itself must have retained the previous value rather
    // than being nulled.
    expect(settingsModule.getCloudKeyStatus()).toEqual({
      gemini_key_set: false,
      groq_key_set: true,
    });
  });
});

describe("loadSettings — a later re-call's cloud-status refetch failure also retains, not nulls (Stage 3 fix)", () => {
  it("a second loadSettings() call (e.g. after models.ts's STT-engine change) with a failing refetch keeps the prior cloud status", async () => {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings({ gemini_api_key: "", groq_api_key: "" }));
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const settingsModule = await import("./settings");

    await vi.waitFor(() => {
      expect(settingsModule.getSettings()).not.toBeNull();
    });
    expect(settingsModule.getCloudKeyStatus()).toEqual({ gemini_key_set: false, groq_key_set: true });

    // A later, independent loadSettings() call (mirrors models.ts calling
    // it after an unrelated settings change) whose cloud-status refetch
    // specifically fails this time.
    apiMock.getSettings.mockResolvedValueOnce(buildSettings({ gemini_api_key: "", groq_api_key: "" }));
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new Error("network down"));

    await settingsModule.loadSettings();

    expect(settingsModule.getCloudKeyStatus()).toEqual({
      gemini_key_set: false,
      groq_key_set: true,
    });
  });
});
