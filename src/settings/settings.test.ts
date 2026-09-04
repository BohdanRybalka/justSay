// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BridgeDiagnosis, UserSettings } from "../api";

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

const sawAuthFailureMock = vi.fn<() => boolean>(() => false);
const lastBridgeDiagnosisMock = vi.fn<() => BridgeDiagnosis>(() => ({ kind: "ok" }));

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    api: apiMock,
    levelStream: vi.fn(() => ({ abort: vi.fn() })),
    sawAuthFailure: sawAuthFailureMock,
    lastBridgeDiagnosis: lastBridgeDiagnosisMock,
  };
});

vi.mock("./tabs/models", () => ({
  renderModels: vi.fn(() => () => {}),
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
    meeting_consent_acknowledged: false,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  sawAuthFailureMock.mockReturnValue(false);
  lastBridgeDiagnosisMock.mockReturnValue({ kind: "ok" });
  vi.resetModules();
  document.body.innerHTML = `
    <ul class="sidebar-nav">
      <li><button class="nav-btn active" data-tab="general">General</button></li>
      <li><button class="nav-btn" data-tab="models">Models</button></li>
    </ul>
    <div id="tab-content"></div>
    <span id="backend-status"></span>
  `;
});

describe("saveSettings — cloud-status refetch failure retains, does not null (Stage 3 fix)", () => {
  it("a failed refetch after saving Gemini leaves the untouched env-sourced Groq row rendering as env, not unset", async () => {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings({ gemini_api_key: "", groq_api_key: "" }));
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const settingsModule = await import("./settings");

    await vi.waitFor(() => {
      expect(settingsModule.getSettings()).not.toBeNull();
      expect(document.getElementById("gemini-save")).not.toBeNull();
    });

    expect(document.getElementById("groq-status")!.textContent).toContain("environment");
    expect(document.getElementById("gemini-status")!.textContent).toBe(
      "No key set — cloud STT will fail.",
    );

    apiMock.updateSettings.mockResolvedValueOnce({
      settings: buildSettings({ gemini_api_key: "***", groq_api_key: "" }),
      warning: null,
    });
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new Error("network down"));

    const geminiInput = document.getElementById("gemini-key-input") as HTMLInputElement;
    geminiInput.value = "AIza-real-key";
    geminiInput.dispatchEvent(new Event("input"));
    (document.getElementById("gemini-save") as HTMLButtonElement).click();

    await vi.waitFor(() => {
      expect(document.getElementById("gemini-status")!.textContent).toBe("Key stored.");
    });

    const groqHint = document.getElementById("groq-status")!.textContent ?? "";
    expect(groqHint).not.toContain("No key set");
    expect(groqHint).toContain("environment");

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

    apiMock.getSettings.mockResolvedValueOnce(buildSettings({ gemini_api_key: "", groq_api_key: "" }));
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new Error("network down"));

    await settingsModule.loadSettings();

    expect(settingsModule.getCloudKeyStatus()).toEqual({
      gemini_key_set: false,
      groq_key_set: true,
    });
  });
});


describe("a shortcut the widget stored while this window was open", () => {
  it("survives a tab switch instead of the General tab redrawing the one loaded at open", async () => {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings({ shortcut: "Ctrl+Alt+KeyV" }));
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const settingsModule = await import("./settings");

    await vi.waitFor(() => {
      expect(document.getElementById("btn-shortcut")).not.toBeNull();
    });
    expect(document.getElementById("btn-shortcut")!.textContent).toBe("Ctrl + Alt + V");

    settingsModule.cachePersistedShortcut("Ctrl+Alt+KeyB");

    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="general"]')!.click();

    expect(document.getElementById("btn-shortcut")!.textContent).toBe("Ctrl + Alt + B");
    expect(settingsModule.getSettings()!.shortcut).toBe("Ctrl+Alt+KeyB");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);
  });
});


const backendStatusEl = () => document.getElementById("backend-status")!;

/** Boots settings.ts with /health healthy and /settings rejecting, and waits
 *  for init()'s failure path to have painted. */
async function bootWithFailedSettingsLoad(error: unknown) {
  apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
  apiMock.getSettings.mockRejectedValue(error);
  apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });

  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  await import("./settings");
  await vi.waitFor(() => {
    expect(document.getElementById("tab-content")!.textContent).toContain("Cannot load settings");
  });
  return { consoleError };
}

describe("backend badge — health 200 + settings 401", () => {
  it("reads 'Backend unauthorized', not the green 'Backend', and carries the diagnosis in its title", async () => {
    const { ApiAuthError } = await import("../api");
    sawAuthFailureMock.mockReturnValue(true);
    lastBridgeDiagnosisMock.mockReturnValue({ kind: "bridge-missing" });

    const { consoleError } = await bootWithFailedSettingsLoad(
      new ApiAuthError("Missing or invalid API token", { kind: "bridge-missing" }),
    );

    await vi.waitFor(() => {
      expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    });
    expect(backendStatusEl().textContent).not.toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator error");
    expect(backendStatusEl().getAttribute("title")).toContain("bridge-missing");

    consoleError.mockRestore();
  });

  it("renders each bridge diagnosis distinguishably in the title", async () => {
    const { ApiAuthError } = await import("../api");
    sawAuthFailureMock.mockReturnValue(true);
    lastBridgeDiagnosisMock.mockReturnValue({
      kind: "invoke-failed",
      detail: "command get_backend_token not found",
    });

    const { consoleError } = await bootWithFailedSettingsLoad(
      new ApiAuthError("Missing or invalid API token", { kind: "invoke-failed", detail: "x" }),
    );

    await vi.waitFor(() => {
      expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    });
    expect(backendStatusEl().getAttribute("title")).toContain(
      "invoke-failed: command get_backend_token not found",
    );

    consoleError.mockRestore();
  });

  it("stays green on an open backend that needs no token — no 401 is ever observed", async () => {
    expect("__TAURI_INTERNALS__" in window).toBe(false);
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: true, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const settingsModule = await import("./settings");

    await vi.waitFor(() => {
      expect(settingsModule.getSettings()).not.toBeNull();
      expect(backendStatusEl().textContent).toBe("Backend");
    });
    expect(backendStatusEl().className).toBe("status-indicator online");
    expect(backendStatusEl().hasAttribute("title")).toBe(false);
  });
});

describe("nav clicks after a failed settings load", () => {
  it("paint a panel naming authentication as the cause and mark the clicked button active", async () => {
    const { ApiAuthError } = await import("../api");
    sawAuthFailureMock.mockReturnValue(true);
    lastBridgeDiagnosisMock.mockReturnValue({ kind: "bridge-missing" });

    const { consoleError } = await bootWithFailedSettingsLoad(
      new ApiAuthError("Missing or invalid API token", { kind: "bridge-missing" }),
    );

    const tabContent = document.getElementById("tab-content")!;
    expect(tabContent.textContent).toContain("authenticate");
    expect(tabContent.textContent).toContain("bridge-missing");

    const general = document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="general"]')!;
    const models = document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!;

    models.click();

    expect(tabContent.textContent!.trim()).not.toBe("");
    expect(tabContent.textContent).toContain("authenticate");
    expect(models.classList.contains("active")).toBe(true);
    expect(general.classList.contains("active")).toBe(false);

    general.click();
    expect(general.classList.contains("active")).toBe(true);
    expect(tabContent.textContent).toContain("authenticate");

    consoleError.mockRestore();
  });
});

describe("a backend that answers but fails the settings request", () => {
  it("names the backend's own error instead of claiming it is not responding", async () => {
    const { consoleError } = await bootWithFailedSettingsLoad(
      new Error("HTTP 500: settings store is locked"),
    );

    const tabContent = document.getElementById("tab-content")!;
    expect(tabContent.textContent).toContain("HTTP 500: settings store is locked");
    expect(tabContent.textContent).not.toContain("not responding");
    expect(tabContent.textContent).not.toContain("authenticate");
    expect(backendStatusEl().textContent).toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator online");

    consoleError.mockRestore();
  });
});

describe("backend badge — health 200, then the first settings request 401s", () => {
  it("repaints to unauthorized as soon as the load fails, without waiting for a poll", async () => {
    const { ApiAuthError } = await import("../api");
    let authFailed = false;
    sawAuthFailureMock.mockImplementation(() => authFailed);
    lastBridgeDiagnosisMock.mockReturnValue({ kind: "invoke-timeout" });

    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getSettings.mockImplementation(() => {
      authFailed = true;
      return Promise.reject(new ApiAuthError("Missing or invalid API token", { kind: "invoke-timeout" }));
    });

    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    await import("./settings");
    await vi.waitFor(() => {
      expect(document.getElementById("tab-content")!.textContent).toContain("Cannot load settings");
    });

    expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    expect(backendStatusEl().className).toBe("status-indicator error");
    expect(backendStatusEl().getAttribute("title")).toContain("invoke-timeout");

    consoleError.mockRestore();
  });
});

describe("backend unreachable from the first poll", () => {
  it("states the failure instead of leaving the content pane empty", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);

    expect(tabContent.textContent!.trim()).not.toBe("");
    expect(tabContent.textContent).toContain("Cannot load settings");
    expect(tabContent.textContent).toContain("was not responding");
    expect(tabContent.querySelector("#btn-retry-settings")).not.toBeNull();
    expect(backendStatusEl().textContent).toBe("Backend offline");
    expect(backendStatusEl().className).toBe("status-indicator offline");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    const painted = tabContent.firstElementChild;
    await vi.advanceTimersByTimeAsync(15000);
    expect(tabContent.firstElementChild).toBe(painted);
    expect(tabContent.textContent).toContain("was not responding");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(tabContent.textContent).toContain("was not responding");

    vi.useRealTimers();
    consoleError.mockRestore();
  });

  it("the retry button loads the settings without restarting the app", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.waitFor(() => {
      expect(tabContent.querySelector("#btn-retry-settings")).not.toBeNull();
    });

    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
      llm_mode: "cloud",
    });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: true, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.click();

    await vi.waitFor(() => {
      expect(tabContent.textContent).not.toContain("Cannot load settings");
    });
    expect(document.getElementById("lang-select")).not.toBeNull();
    expect(backendStatusEl().className).toBe("status-indicator online");

    consoleError.mockRestore();
  });

  it("a second retry while one is in flight does not start a second load", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.waitFor(() => {
      expect(tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.disabled).toBe(
        false,
      );
    });

    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
      llm_mode: "cloud",
    });
    let release!: (settings: UserSettings) => void;
    apiMock.getSettings.mockReturnValue(
      new Promise<UserSettings>((resolve) => (release = resolve)),
    );
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: true, groq_key_set: true });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.click();
    await vi.waitFor(() => {
      expect(apiMock.getSettings).toHaveBeenCalledTimes(2);
    });

    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    const repainted = tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!;
    expect(repainted.disabled).toBe(true);
    repainted.click();

    release(buildSettings());
    await vi.waitFor(() => {
      expect(tabContent.textContent).not.toContain("Cannot load settings");
    });
    expect(apiMock.getSettings).toHaveBeenCalledTimes(2);

    consoleError.mockRestore();
  });

  it("a backend that accepts the connection and never answers still offers a way out", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
      llm_mode: "cloud",
    });
    apiMock.getSettings.mockReturnValue(new Promise(() => {}));
    apiMock.cloudKeyStatus.mockReturnValue(new Promise(() => {}));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(tabContent.textContent).toContain("Loading settings");
    expect(tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.disabled).toBe(true);

    await vi.advanceTimersByTimeAsync(41_000);

    expect(tabContent.textContent).toContain("Cannot load settings");
    expect(tabContent.textContent).toContain("Loading settings did not finish in time");
    expect(tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.disabled).toBe(false);

    vi.useRealTimers();
    consoleError.mockRestore();
  });

  it("names the endpoint and the budget when the budget that expired knows them", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const { TimedOutError } = await import("../timeout");
    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
      llm_mode: "cloud",
    });
    apiMock.getSettings.mockRejectedValue(new TimedOutError(15_000, "/settings"));
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;
    await vi.advanceTimersByTimeAsync(0);

    expect(tabContent.textContent).toContain("Cannot load settings");
    expect(tabContent.textContent).toContain("(/settings, 15 s)");
    expect(tabContent.textContent).not.toContain("accepted");

    vi.useRealTimers();
    consoleError.mockRestore();
  });

  it("a retry that fails again leaves the button usable rather than stuck", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.waitFor(() => {
      expect(tabContent.querySelector("#btn-retry-settings")).not.toBeNull();
    });
    tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.click();

    await vi.waitFor(() => {
      expect(apiMock.getSettings).toHaveBeenCalledTimes(2);
    });
    await vi.waitFor(() => {
      const again = tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!;
      expect(again.disabled).toBe(false);
      expect(again.textContent).toBe("Try again");
    });

    consoleError.mockRestore();
  });
});

describe("a 401 observed after settings have loaded", () => {
  it("un-latches on a later poll, and no poll ever writes the rendered tab", async () => {
    vi.useFakeTimers();
    let authFailed = false;
    sawAuthFailureMock.mockImplementation(() => authFailed);
    lastBridgeDiagnosisMock.mockReturnValue({ kind: "invoke-failed", detail: "boom" });

    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
    apiMock.getSettings.mockResolvedValue(buildSettings());

    const settingsModule = await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(settingsModule.getSettings()).not.toBeNull();
    expect(backendStatusEl().textContent).toBe("Backend");

    const keyInput = document.getElementById("gemini-key-input") as HTMLInputElement;
    keyInput.value = "typing-in-progress";

    authFailed = true;

    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    expect(backendStatusEl().className).toBe("status-indicator error");
    expect(backendStatusEl().getAttribute("title")).toContain("invoke-failed: boom");
    expect(tabContent.textContent).not.toContain("Cannot load settings");
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(document.getElementById("gemini-key-input")).toBe(keyInput);
    expect(keyInput.value).toBe("typing-in-progress");

    authFailed = false;

    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator online");
    expect(backendStatusEl().hasAttribute("title")).toBe(false);
    expect(tabContent.textContent).not.toContain("Cannot load settings");
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(document.getElementById("gemini-key-input")).toBe(keyInput);
    expect(keyInput.value).toBe("typing-in-progress");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });
});

describe("a settings load that has not settled", () => {
  it("keeps saying it is waiting while the badge goes on tracking /health", async () => {
    vi.useFakeTimers();
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getSettings.mockImplementation(() => new Promise<UserSettings>(() => {}));

    const settingsModule = await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(tabContent.textContent).toContain("Loading settings");
    expect(tabContent.textContent).toContain("Waiting for the backend to answer");
    expect(backendStatusEl().textContent).toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator online");

    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend offline");
    expect(tabContent.textContent).toContain("Loading settings");

    await vi.advanceTimersByTimeAsync(15000);
    expect(settingsModule.getSettings()).toBeNull();
    expect(tabContent.textContent).toContain("Loading settings");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });
});

describe("a settings load that fails after the backend has gone away", () => {
  it("writes its sentence from its own probe, even when a later poll supersedes it", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const pending: Array<(ok: boolean) => void> = [];
    apiMock.health.mockImplementation(
      () =>
        new Promise((resolve, reject) => {
          pending.push((ok) =>
            ok
              ? resolve({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" })
              : reject(new TypeError("Failed to fetch")),
          );
        }),
    );
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(5000);
    expect(pending.length).toBe(2);

    pending[1](true);
    await vi.advanceTimersByTimeAsync(0);
    expect(backendStatusEl().textContent).toBe("Backend");

    pending[0](false);
    await vi.advanceTimersByTimeAsync(0);

    expect(tabContent.textContent).toContain("was not responding");
    expect(tabContent.textContent).not.toContain("The backend answered");
    expect(backendStatusEl().textContent).toBe("Backend");

    vi.useRealTimers();
    consoleError.mockRestore();
  });
});

describe("the Settings window's own health poll", () => {
  it("keeps probing while one is unanswered, and lets only the newest answer repaint", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const pending: Array<(ok: boolean) => void> = [];
    apiMock.health.mockImplementation(
      () =>
        new Promise((resolve, reject) => {
          pending.push((ok) =>
            ok
              ? resolve({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" })
              : reject(new TypeError("Failed to fetch")),
          );
        }),
    );
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    await vi.advanceTimersByTimeAsync(0);
    const before = apiMock.health.mock.calls.length;

    await vi.advanceTimersByTimeAsync(5000 * 4);

    expect(apiMock.health.mock.calls.length - before).toBe(4);

    pending[pending.length - 1](true);
    await vi.advanceTimersByTimeAsync(0);
    for (const settle of pending.slice(0, -1)) settle(false);
    await vi.advanceTimersByTimeAsync(0);

    expect(backendStatusEl().textContent).toBe("Backend");

    vi.useRealTimers();
    consoleError.mockRestore();
  });

  it("the retry button probes rather than joining a probe already in flight", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;
    await vi.advanceTimersByTimeAsync(0);
    await vi.waitFor(() => {
      expect(tabContent.querySelector("#btn-retry-settings")).not.toBeNull();
    });

    apiMock.health.mockImplementation(() => new Promise(() => {}));
    await vi.advanceTimersByTimeAsync(5000);
    const before = apiMock.health.mock.calls.length;

    tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.health.mock.calls.length - before).toBe(1);

    vi.useRealTimers();
    consoleError.mockRestore();
  });
});
