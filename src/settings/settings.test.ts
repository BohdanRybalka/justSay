// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BridgeDiagnosis, UserSettings } from "../api";

const apiMock = {
  health: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
  cloudKeyStatus: vi.fn(),
  getStorageInfo: vi.fn(),
  audioDiscard: vi.fn(),
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

const modelsTab = {
  destroy: vi.fn(),
  releaseResources: vi.fn(),
  resumeResources: vi.fn(),
};

const modelsMountedHidden: boolean[] = [];

vi.mock("./tabs/models", () => ({
  renderModels: vi.fn((container: HTMLElement, _settings: unknown, windowHidden: boolean) => {
    modelsMountedHidden.push(windowHidden);
    container.innerHTML = '<div id="models-tab-body"></div>';
    return modelsTab;
  }),
}));

const eventListeners = new Map<string, (event: unknown) => unknown>();

const listenAttemptsByEvent = new Map<string, number>();
const refusedEvents = new Set<string>();

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (payload: unknown) => unknown) => {
    listenAttemptsByEvent.set(event, (listenAttemptsByEvent.get(event) ?? 0) + 1);
    if (refusedEvents.has(event)) {
      throw new Error(`the event bus refused a subscription to ${event}`);
    }
    eventListeners.set(event, handler);
    return () => {};
  }),
  emit: vi.fn(async () => {}),
}));

/** How often this page has asked the bus for one named subscription.
 *
 *  Keyed by name rather than totalled, because the General tab subscribes to
 *  `shortcut-applied` once the settings load resolves and a total cannot tell
 *  that attempt from the two this suite steers. */
function listenAttempts(event: string): number {
  return listenAttemptsByEvent.get(event) ?? 0;
}

let windowIsVisible: boolean | null = null;
let visibilityReads = 0;
let holdVisibilityRead = false;
let releaseVisibilityRead: () => void = () => {};

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    isVisible: async () => {
      visibilityReads += 1;
      if (holdVisibilityRead) {
        await new Promise<void>((resolve) => (releaseVisibilityRead = resolve));
      }
      if (windowIsVisible === null) throw new Error("no window to ask outside Tauri");
      return windowIsVisible;
    },
  }),
}));

/** Delivers the first `settings-shown`, the way opening the hidden window does.
 *
 *  The shell creates the Settings window invisible, so a page that has just
 *  booted believes it is hidden and drops a dismissal it never saw paired. */
async function openSettingsWindow(): Promise<void> {
  const { EVENT_SETTINGS_SHOWN } = await import("../contracts");
  await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_SHOWN)).toBeTypeOf("function"));
  await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});
}

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
    ...overrides,
  };
}

beforeEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  sawAuthFailureMock.mockReturnValue(false);
  lastBridgeDiagnosisMock.mockReturnValue({ kind: "ok" });
  vi.resetModules();
  eventListeners.clear();
  listenAttemptsByEvent.clear();
  refusedEvents.clear();
  windowIsVisible = null;
  visibilityReads = 0;
  holdVisibilityRead = false;
  releaseVisibilityRead = () => {};
  modelsMountedHidden.length = 0;
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
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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
  apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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

    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
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

    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
    apiMock.getSettings.mockResolvedValue(buildSettings());

    const settingsModule = await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(settingsModule.getSettings()).not.toBeNull();
    expect(backendStatusEl().textContent).toBe("Backend");
    await openSettingsWindow();

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
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getSettings.mockImplementation(() => new Promise<UserSettings>(() => {}));

    const settingsModule = await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(tabContent.textContent).toContain("Loading settings");
    expect(tabContent.textContent).toContain("Waiting for the backend to answer");
    expect(backendStatusEl().textContent).toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator online");
    await openSettingsWindow();

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
              ? resolve({ status: "ok", version: "0.0.0", stt_mode: "cloud" })
              : reject(new TypeError("Failed to fetch")),
          );
        }),
    );
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;
    await vi.advanceTimersByTimeAsync(0);
    await openSettingsWindow();

    await vi.advanceTimersByTimeAsync(5000);
    expect(pending.length).toBe(3);

    pending[2](true);
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
              ? resolve({ status: "ok", version: "0.0.0", stt_mode: "cloud" })
              : reject(new TypeError("Failed to fetch")),
          );
        }),
    );
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    await vi.advanceTimersByTimeAsync(0);
    await openSettingsWindow();
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
    await openSettingsWindow();

    apiMock.health.mockImplementation(() => new Promise(() => {}));
    await vi.advanceTimersByTimeAsync(5000);
    const before = apiMock.health.mock.calls.length;

    tabContent.querySelector<HTMLButtonElement>("#btn-retry-settings")!.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.health.mock.calls.length - before).toBe(1);

    vi.useRealTimers();
    consoleError.mockRestore();
  });

  it("stays silent until the window is opened for the first time", async () => {
    vi.useFakeTimers();
    try {
      apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
      apiMock.getSettings.mockResolvedValue(buildSettings());
      apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
      apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

      const { EVENT_SETTINGS_SHOWN } = await import("../contracts");
      await import("./settings");
      await vi.waitFor(() =>
        expect(eventListeners.get(EVENT_SETTINGS_SHOWN)).toBeTypeOf("function"),
      );
      const beforeAnyShow = apiMock.health.mock.calls.length;

      await vi.advanceTimersByTimeAsync(30_000);

      expect(
        apiMock.health.mock.calls.length,
        "the window is created invisible, so probing every 5 s from launch is the " +
          "same waste as probing every 5 s after a dismissal",
      ).toBe(beforeAnyShow);

      await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});

      expect(
        apiMock.health.mock.calls.length,
        "the badge a user reads on the first open must not wait out a whole interval",
      ).toBe(beforeAnyShow + 1);

      await vi.advanceTimersByTimeAsync(5000);
      expect(apiMock.health.mock.calls.length).toBe(beforeAnyShow + 2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("the Settings window being dismissed", () => {
  it("releases the microphone the General tab was holding", async () => {
    const { EVENT_SETTINGS_HIDDEN } = await import("../contracts");
    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
    });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
    apiMock.audioStart.mockResolvedValue({
      is_recording: true,
      duration_seconds: 0,
      level_db: -60,
      session_id: null,
    });
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 2 });

    await import("./settings");
    await vi.waitFor(() => expect(document.getElementById("btn-test-mic")).not.toBeNull());
    await openSettingsWindow();

    const button = document.getElementById("btn-test-mic") as HTMLButtonElement;
    button.click();
    await vi.waitFor(() => expect(apiMock.audioStart).toHaveBeenCalledOnce());

    await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_HIDDEN)).toBeTypeOf("function"));
    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});

    await vi.waitFor(() =>
      expect(apiMock.audioDiscard).toHaveBeenCalledWith(apiMock.audioStart.mock.calls[0][0]),
    );
  });

  it("releases without re-mounting the tab, so its reads do not run again while it is invisible", async () => {
    const { EVENT_SETTINGS_HIDDEN } = await import("../contracts");
    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0.0.0",
      stt_mode: "cloud",
    });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    await import("./settings");
    await vi.waitFor(() => expect(document.getElementById("btn-test-mic")).not.toBeNull());
    await openSettingsWindow();
    const readsBefore = apiMock.getStorageInfo.mock.calls.length;
    const button = document.getElementById("btn-test-mic") as HTMLButtonElement;

    await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_HIDDEN)).toBeTypeOf("function"));
    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});
    await vi.waitFor(() => expect(apiMock.cloudKeyStatus).toHaveBeenCalled());

    expect(apiMock.getStorageInfo.mock.calls.length).toBe(readsBefore);
    expect(document.getElementById("btn-test-mic")).toBe(button);
  });

  it("asks for nothing at all when no tab was ever mounted, because no settings loaded", async () => {
    const { EVENT_SETTINGS_HIDDEN } = await import("../contracts");
    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.getSettings.mockRejectedValue(new TypeError("Failed to fetch"));
    apiMock.cloudKeyStatus.mockRejectedValue(new TypeError("Failed to fetch"));

    await import("./settings");
    await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_HIDDEN)).toBeTypeOf("function"));
    await openSettingsWindow();

    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});

    expect(apiMock.getStorageInfo).not.toHaveBeenCalled();
    expect(apiMock.audioDiscard).not.toHaveBeenCalled();
  });
});

describe("a file dropped where nothing in the page handles it", () => {
  async function bootSettingsWindow(): Promise<void> {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    await import("./settings");
    await vi.waitFor(() => expect(document.getElementById("btn-test-mic")).not.toBeNull());
  }

  function dispatchOnTheNav(type: string, carried: string[]): Event {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: { types: carried } });
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.dispatchEvent(event);
    return event;
  }

  it("is swallowed, so the webview cannot navigate away from the Settings UI", async () => {
    await bootSettingsWindow();

    const dragover = dispatchOnTheNav("dragover", ["Files"]);
    const drop = dispatchOnTheNav("drop", ["Files"]);

    expect(
      dragover.defaultPrevented,
      "without a prevented dragover the webview refuses the drag outright (ADR 087)",
    ).toBe(true);
    expect(
      drop.defaultPrevented,
      "an unprevented drop is a browser navigation to the dropped file, which replaces " +
        "the whole Settings UI (ADR 087)",
    ).toBe(true);
  });

  it("leaves a text drag alone, so a dragged folder path still lands in a field", async () => {
    await bootSettingsWindow();

    const dragover = dispatchOnTheNav("dragover", ["text/plain"]);
    const drop = dispatchOnTheNav("drop", ["text/plain"]);

    expect(
      dragover.defaultPrevented,
      "cancelling a text dragover shows a copy cursor over every tab that takes no file",
    ).toBe(false);
    expect(
      drop.defaultPrevented,
      "cancelling a text drop stops the browser inserting a folder path dragged from " +
        "Explorer into the output-directory field, and nothing puts it there instead",
    ).toBe(false);
  });
});

describe("the Settings window coming back after a dismissal", () => {
  async function bootOnTheModelsTab(): Promise<void> {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    await import("./settings");
    await vi.waitFor(() => expect(document.getElementById("btn-test-mic")).not.toBeNull());
    await openSettingsWindow();
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
  }

  it("touches no tab hook on a show that followed no dismissal", async () => {
    const { EVENT_SETTINGS_SHOWN } = await import("../contracts");
    await bootOnTheModelsTab();
    await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_SHOWN)).toBeTypeOf("function"));

    await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});

    expect(
      modelsTab.resumeResources,
      "the tray item shows an already visible window, and resuming there starts a " +
        "second interval beside the live one",
    ).not.toHaveBeenCalled();
    expect(modelsTab.releaseResources).not.toHaveBeenCalled();
  });

  it("mounts a tab into a dismissed window without starting anything, then resumes it once", async () => {
    let releaseSettings: (loaded: UserSettings) => void = () => {};
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockImplementation(
      () => new Promise<UserSettings>((resolve) => (releaseSettings = resolve)),
    );
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
    await import("./settings");
    await openSettingsWindow();

    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});

    releaseSettings(buildSettings());
    await vi.waitFor(() => expect(document.getElementById("models-tab-body")).not.toBeNull());

    expect(
      modelsTab.releaseResources,
      "a slow cold start lands the tab in a window the user has already closed, and a tab " +
        "that keeps its interval there polls for the life of the app",
    ).toHaveBeenCalledTimes(1);
    expect(modelsTab.resumeResources).not.toHaveBeenCalled();

    await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});

    expect(
      modelsTab.resumeResources,
      "and the show that follows must leave one live interval, not a second beside it",
    ).toHaveBeenCalledTimes(1);
    expect(modelsTab.releaseResources).toHaveBeenCalledTimes(1);
  });

  it("tells each tab whether the window it is mounting into is dismissed", async () => {
    let releaseSettings: (loaded: UserSettings) => void = () => {};
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockImplementation(
      () => new Promise<UserSettings>((resolve) => (releaseSettings = resolve)),
    );
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
    await import("./settings");
    await openSettingsWindow();

    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});

    releaseSettings(buildSettings());
    await vi.waitFor(() => expect(modelsMountedHidden).toHaveLength(1));

    expect(
      modelsMountedHidden[0],
      "a release only runs once the tab has returned, by which point its mount-time " +
        "reads are away; a tab that is told can skip them instead",
    ).toBe(true);

    await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="general"]')!.click();
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();

    expect(
      modelsMountedHidden[modelsMountedHidden.length - 1],
      "and a tab mounted into a window the user is looking at must read at once",
    ).toBe(false);
  });

  it("resumes the tab once, however often the shell repeats either edge", async () => {
    const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
    await bootOnTheModelsTab();
    await vi.waitFor(() => expect(eventListeners.get(EVENT_SETTINGS_SHOWN)).toBeTypeOf("function"));

    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});
    await eventListeners.get(EVENT_SETTINGS_HIDDEN)!({});
    await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});
    await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});

    expect(modelsTab.releaseResources).toHaveBeenCalledTimes(1);
    expect(modelsTab.resumeResources).toHaveBeenCalledTimes(1);
  });

  it("polls on its own when the event bus refuses every subscription", async () => {
    vi.useFakeTimers();
    try {
      const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
      refusedEvents.add(EVENT_SETTINGS_SHOWN);
      refusedEvents.add(EVENT_SETTINGS_HIDDEN);
      apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
      apiMock.getSettings.mockResolvedValue(buildSettings());
      apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
      apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

      await import("./settings");
      await vi.waitFor(() => expect(listenAttempts(EVENT_SETTINGS_SHOWN)).toBe(1));

      const before = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "a page that attached nothing hears no show ever, so believing the window hidden " +
          "would leave the settings screen served by Vite reporting the backend as unknown " +
          "for the life of the app",
      ).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps the resume when the event bus accepts only one subscription", async () => {
    vi.useFakeTimers();
    try {
      const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
      refusedEvents.add(EVENT_SETTINGS_HIDDEN);
      apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
      apiMock.getSettings.mockResolvedValue(buildSettings());
      apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
      apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

      await import("./settings");
      await vi.waitFor(() => expect(listenAttempts(EVENT_SETTINGS_HIDDEN)).toBe(1));

      expect(
        eventListeners.has(EVENT_SETTINGS_SHOWN),
        "the subscription that survives a partial attach has to be the one that starts " +
          "the polling again; a window left holding only the release freezes its badge " +
          "at the first close and never recovers",
      ).toBe(true);
      expect(eventListeners.has(EVENT_SETTINGS_HIDDEN)).toBe(false);

      const before = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "a page that did attach the show edge still hears the window arrive, so assuming " +
          "it is already visible would poll an invisible window with no edge left to stop it",
      ).toBe(before);

      await eventListeners.get(EVENT_SETTINGS_SHOWN)!({});
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "and the show it can still hear is what starts the polling",
      ).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("the Settings window's own health poll across a dismissal", () => {
  async function bootUnderFakeTimers(): Promise<{
    hidden: (payload: unknown) => unknown;
    shown: (payload: unknown) => unknown;
  }> {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
    await import("./settings");
    await openSettingsWindow();
    return {
      hidden: eventListeners.get(EVENT_SETTINGS_HIDDEN)!,
      shown: eventListeners.get(EVENT_SETTINGS_SHOWN)!,
    };
  }

  it("goes quiet for as long as the window is hidden and probes at once on the way back", async () => {
    vi.useFakeTimers();
    try {
      const { hidden, shown } = await bootUnderFakeTimers();

      await hidden({});
      const whileHidden = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(30_000);

      expect(
        apiMock.health.mock.calls.length,
        "a hidden window that keeps probing is the request every 5 s this spec closes",
      ).toBe(whileHidden);

      await shown({});

      expect(
        apiMock.health.mock.calls.length,
        "the badge a returning user reads must not wait out a whole interval",
      ).toBe(whileHidden + 1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("leaves exactly one live probe after a repeated show", async () => {
    vi.useFakeTimers();
    try {
      const { hidden, shown } = await bootUnderFakeTimers();

      await hidden({});
      await shown({});
      await shown({});
      const afterResume = apiMock.health.mock.calls.length;

      await vi.advanceTimersByTimeAsync(5000);
      expect(apiMock.health.mock.calls.length).toBe(afterResume + 1);

      await vi.advanceTimersByTimeAsync(5000);
      expect(apiMock.health.mock.calls.length).toBe(afterResume + 2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("lets no probe that answers after the dismissal repaint the badge", async () => {
    vi.useFakeTimers();
    try {
      const { hidden } = await bootUnderFakeTimers();
      const badge = document.getElementById("backend-status")!;
      await vi.waitFor(() => expect(badge.textContent).toBe("Backend"));

      let failTheProbe: (reason: Error) => void = () => {};
      apiMock.health.mockImplementationOnce(
        () => new Promise((_resolve, reject) => (failTheProbe = reject)),
      );
      await vi.advanceTimersByTimeAsync(5000);

      await hidden({});
      failTheProbe(new Error("the backend went away"));
      await vi.advanceTimersByTimeAsync(0);

      expect(
        badge.textContent,
        "a probe the dismissal did not disown lands on an invisible window and repaints it",
      ).toBe("Backend");
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("a show the Settings page was not yet listening for", () => {
  function mockABackendThatAnswers(): void {
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud" });
    apiMock.getSettings.mockResolvedValue(buildSettings());
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
  }

  it("picks the window up from its own visibility when the show arrived first", async () => {
    vi.useFakeTimers();
    try {
      windowIsVisible = true;
      mockABackendThatAnswers();

      await import("./settings");
      await vi.waitFor(() => expect(visibilityReads).toBe(1));

      const before = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "the tray can show the window before this page subscribed, and a page that waits " +
          "for the next announcement believes itself hidden for the whole first open",
      ).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stays quiet when the window itself reports it is not up yet", async () => {
    vi.useFakeTimers();
    try {
      windowIsVisible = false;
      mockABackendThatAnswers();

      await import("./settings");
      await vi.waitFor(() => expect(visibilityReads).toBe(1));

      const before = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(30_000);

      expect(
        apiMock.health.mock.calls.length,
        "reading the window must not become a second way to start polling one nobody opened",
      ).toBe(before);
    } finally {
      vi.useRealTimers();
    }
  });

  it("drops a reading the shell's own announcement overtook", async () => {
    vi.useFakeTimers();
    try {
      holdVisibilityRead = true;
      windowIsVisible = false;
      mockABackendThatAnswers();

      await import("./settings");
      await vi.waitFor(() => expect(visibilityReads).toBe(1));
      await openSettingsWindow();
      const afterShow = apiMock.health.mock.calls.length;

      releaseVisibilityRead();
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "the tray showed the window while the read was away, so the reading is older " +
          "than the announcement; taking it for a dismissal stops the probe on a window " +
          "the user is looking at, and the next hide dedupes so nothing recovers it",
      ).toBeGreaterThan(afterShow);
    } finally {
      vi.useRealTimers();
    }
  });

  it("polls when it could attach no listener, whatever the window reports", async () => {
    vi.useFakeTimers();
    try {
      const { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } = await import("../contracts");
      refusedEvents.add(EVENT_SETTINGS_SHOWN);
      refusedEvents.add(EVENT_SETTINGS_HIDDEN);
      windowIsVisible = false;
      mockABackendThatAnswers();

      await import("./settings");
      await vi.waitFor(() => expect(listenAttempts(EVENT_SETTINGS_SHOWN)).toBe(1));

      const before = apiMock.health.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5000);

      expect(
        apiMock.health.mock.calls.length,
        "a page holding no show edge can never be told the window arrived, so believing " +
          "the reading leaves it with no listener, no interval and no edge left to start " +
          "one for the life of the app",
      ).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
    }
  });
});
