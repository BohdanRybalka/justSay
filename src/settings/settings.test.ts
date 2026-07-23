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

// Only the api *object* and the two token-state readers are replaced; the rest
// of ../api (notably the real ApiAuthError class) is passed through, so the
// `instanceof` branches under test run against the real type rather than a
// stand-in. Whether the real sawAuthFailure() flips correctly is api.ts's own
// contract and is covered in src/api.test.ts.
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
  vi.useRealTimers();
  vi.clearAllMocks();
  sawAuthFailureMock.mockReturnValue(false);
  lastBridgeDiagnosisMock.mockReturnValue({ kind: "ok" });
  // Each test dynamically re-imports "./settings" fresh (below), so the
  // module registry must be reset too -- otherwise a second test would
  // reuse the first test's module instance, whose captured `tabContent`
  // element reference is now detached from the DOM rebuilt here.
  vi.resetModules();
  // The nav buttons are part of the real index.html markup; switchTab() reads
  // them, and the "no click is a no-op" guarantee is about these elements.
  //
  // #backend-status is deliberately bare, and must stay that way even though
  // index.html:33 ships `class="status-indicator offline">Backend offline`.
  // That shipped default is character-for-character what the offline
  // assertions below expect, so seeding it makes them pass against an element
  // no code ever wrote — a mutation suppressing the offline paint entirely
  // left this whole file green (review iteration 4, finding 3). The rule is
  // general: no fixture may seed a value an assertion in the same file expects
  // as an outcome. An empty element means "nothing has painted this yet",
  // which is the only honest starting state for a test of what paints it.
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

// --- Spec 042: an authenticated-but-refused backend must not read as healthy ---
//
// `GET /health` is exempt from the backend's token gate, so a WebView that
// never got its token gets 200 there and 401 everywhere else. That combination
// painted a green "Backend" badge over a window where nothing worked.

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
    // The plain-browser dev flow: no Tauri bridge in the page at all, and a
    // manually launched backend that returns 200 for everything. The new auth
    // axis must not fire here.
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

    // Clicking the already-active tab is also not a no-op.
    general.click();
    expect(general.classList.contains("active")).toBe(true);
    expect(tabContent.textContent).toContain("authenticate");

    consoleError.mockRestore();
  });
});

// AC 13. "Backend is not responding" is a claim about the backend, and it must
// not be made about a backend that answered. This case used to pin the old
// single-branch wording; a /health 200 with a /settings 500 put a green badge
// and a "not responding" panel in the same window, one of them false.
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

// Matrix row 3, the real boot sequence — not the AC 3 test above, which seeds
// sawAuthFailure()=true before the module loads and so never exercises t=0.
// Here the flag is false during init()'s first checkBackend() and flips true
// only when the first gated request 401s, exactly as api.ts's real
// recordAuthOutcome() does. The badge must read unauthorized the instant the
// load fails, not one poll later — asserted synchronously off the panel render,
// so it cannot pass by waiting out a timer.
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

    // Synchronous: the panel render and the badge repaint are in the same catch,
    // so once the panel is present the badge has been repainted too. Without the
    // catch-block repaint the badge is still the green "Backend" here.
    expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    expect(backendStatusEl().className).toBe("status-indicator error");
    expect(backendStatusEl().getAttribute("title")).toContain("invoke-timeout");

    consoleError.mockRestore();
  });
});

// AC 11, matrix row 6. `#tab-content` ships empty, so a window that gives up
// on an unreachable backend without saying so is the exact symptom this spec is
// named after, rebuilt from the other side. Both halves must be falsifiable —
// the badge half only is because the fixture is bare (see beforeEach).
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
    expect(tabContent.textContent).toContain("restart JustSay");
    expect(backendStatusEl().textContent).toBe("Backend offline");
    expect(backendStatusEl().className).toBe("status-indicator offline");
    // The load is issued unconditionally, even behind a failed /health: its own
    // rejection is a truer error than one inferred from /health's. Exactly one
    // attempt — nothing retries it.
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    // Three further polls must neither blank the panel nor rebuild its DOM, and
    // must not start a second load. Node identity is the check, since the copy
    // is unchanged either way.
    const painted = tabContent.firstElementChild;
    await vi.advanceTimersByTimeAsync(15000);
    expect(tabContent.firstElementChild).toBe(painted);
    expect(tabContent.textContent).toContain("was not responding");
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    // And a nav click in that state must not claim something is still loading.
    document.querySelector<HTMLButtonElement>('.nav-btn[data-tab="models"]')!.click();
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(tabContent.textContent).toContain("was not responding");

    vi.useRealTimers();
    consoleError.mockRestore();
  });
});

// AC 12, matrix row 2. The mirror of the original defect — a red badge latched
// over a window whose tabs all work — plus the rule that keeps the two surfaces
// from disagreeing: the badge tracks the most recent gated outcome every 5 s,
// and no timer touches `#tab-content` at all.
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

    // The user is mid-edit in the rendered General tab...
    const keyInput = document.getElementById("gemini-key-input") as HTMLInputElement;
    keyInput.value = "typing-in-progress";

    // ...and some other gated request — the widget's dictation, say — comes
    // back 401. The poll observes it on the badge without touching the pane.
    authFailed = true;

    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend unauthorized");
    expect(backendStatusEl().className).toBe("status-indicator error");
    expect(backendStatusEl().getAttribute("title")).toContain("invoke-failed: boom");
    expect(tabContent.textContent).not.toContain("Cannot load settings");
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(document.getElementById("gemini-key-input")).toBe(keyInput);
    expect(keyInput.value).toBe("typing-in-progress");

    // A later gated request succeeds — a save, or opening History. api.ts
    // clears the flag; the badge follows within one poll.
    authFailed = false;

    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend");
    expect(backendStatusEl().className).toBe("status-indicator online");
    expect(backendStatusEl().hasAttribute("title")).toBe(false);
    expect(tabContent.textContent).not.toContain("Cannot load settings");
    expect(tabContent.textContent).not.toContain("Loading settings");
    expect(document.getElementById("gemini-key-input")).toBe(keyInput);
    expect(keyInput.value).toBe("typing-in-progress");
    // One load for the life of the module — nothing reloads on a timer.
    expect(apiMock.getSettings).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });
});

// Matrix rows 4 and 10 — the two states where nothing has settled. Row 4 is the
// ordinary first second of startup; row 10 is a `GET /settings` that connects
// and never answers, which request() does not bound (deferred on purpose). Both
// must say they are waiting rather than show a blank pane, and the badge must
// keep tracking /health regardless, since the interval is armed before the load.
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

    // The backend goes away while the load is still suspended. The badge is the
    // only surface that may move — the pane is the record of an attempt that
    // has not finished, and no timer writes it.
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

// The defect both iteration 3 and iteration 4 graded MAJOR was one shape: a
// value read before an `await` and acted on after it. Here the backend is
// reachable when the load starts and gone by the time it fails, so a captured
// reachability picks branch 3 ("the backend answered") over the branch the
// freshest measurement demands. Capturing `backendReachable` before
// `await loadSettings()` fails this case.
describe("a settings load that fails after the backend has gone away", () => {
  it("reports the reachability the latest poll measured, not the one at load time", async () => {
    vi.useFakeTimers();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    apiMock.health.mockResolvedValue({ status: "ok", version: "0.0.0", stt_mode: "cloud", llm_mode: "cloud" });
    apiMock.cloudKeyStatus.mockResolvedValue({ gemini_key_set: false, groq_key_set: false });

    let failSettings!: (error: unknown) => void;
    apiMock.getSettings.mockImplementation(
      () => new Promise<UserSettings>((_resolve, reject) => { failSettings = reject; }),
    );

    await import("./settings");
    const tabContent = document.getElementById("tab-content")!;

    await vi.advanceTimersByTimeAsync(0);
    expect(backendStatusEl().textContent).toBe("Backend");

    apiMock.health.mockRejectedValue(new TypeError("Failed to fetch"));
    await vi.advanceTimersByTimeAsync(5000);
    expect(backendStatusEl().textContent).toBe("Backend offline");

    failSettings(new TypeError("Failed to fetch"));
    await vi.advanceTimersByTimeAsync(0);

    expect(tabContent.textContent).toContain("was not responding");
    expect(tabContent.textContent).not.toContain("The backend answered");

    vi.useRealTimers();
    consoleError.mockRestore();
  });
});
