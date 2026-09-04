// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SHORTCUT } from "../../accelerator";
import type { UserSettings } from "../../api";
import { TimedOutError } from "../../timeout";

const apiMock = {
  getStorageInfo: vi.fn(),
  audioStop: vi.fn(),
  audioStatus: vi.fn(),
  audioStart: vi.fn(),
  cleanupTemp: vi.fn(),
  updateSettings: vi.fn(),
};

const levelStreamMock = vi.fn();

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: apiMock, levelStream: levelStreamMock };
});

const saveSettingsMock = vi.fn();
const getCloudKeyStatusMock = vi.fn();
const cachePersistedShortcutMock = vi.fn();

vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
  getCloudKeyStatus: getCloudKeyStatusMock,
  cachePersistedShortcut: cachePersistedShortcutMock,
}));

vi.mock("./keys", () => ({
  renderKeys: vi.fn(() => () => {}),
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", () => ({
  notifyError: notifyErrorMock,
}));

interface ShortcutAppliedEvent {
  payload: {
    shortcut: string;
    ok: boolean;
    reason: string | null;
    persisted: boolean | null;
    stillActive: string | null;
  };
}

const emitMock = vi.fn();
const unlistenMock = vi.fn();
const listenMock = vi.fn(
  async (_event: string, _handler: (event: ShortcutAppliedEvent) => void) => unlistenMock,
);
vi.mock("@tauri-apps/api/event", () => ({
  emit: emitMock,
  listen: listenMock,
}));

const checkMock = vi.fn();
vi.mock("@tauri-apps/plugin-updater", () => ({
  check: checkMock,
}));

const relaunchMock = vi.fn();
vi.mock("@tauri-apps/plugin-process", () => ({
  relaunch: relaunchMock,
}));

vi.mock("@tauri-apps/api/app", () => ({
  getVersion: vi.fn(async () => "0.13.0"),
}));

const { renderGeneral } = await import("./general");
const { REQUEST_TIMEOUT_MS } = await import("../../api");

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: DEFAULT_SHORTCUT,
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

const consoleErrorMock = vi.fn();

beforeEach(() => {
  vi.resetAllMocks();
  vi.spyOn(console, "error").mockImplementation(consoleErrorMock);
  listenMock.mockImplementation(async () => unlistenMock);
  apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });
  levelStreamMock.mockImplementation(() => ({ abort: vi.fn() }));
});

describe("renderGeneral — the updates button", () => {
  function renderUpdates(): {
    button: HTMLButtonElement;
    status: HTMLElement;
  } {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    return {
      button: container.querySelector<HTMLButtonElement>("#btn-check-updates")!,
      status: container.querySelector<HTMLElement>("#updates-status")!,
    };
  }

  function buildUpdate(downloadAndInstall = vi.fn(async () => {})) {
    return { version: "0.14.0", currentVersion: "0.13.0", downloadAndInstall };
  }

  it("one click on Install & Restart installs once and checks nothing", async () => {
    const downloadAndInstall = vi.fn(async () => {});
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Install & Restart");
    });
    expect(checkMock).toHaveBeenCalledTimes(1);

    button.click();
    await vi.waitFor(() => {
      expect(relaunchMock).toHaveBeenCalledTimes(1);
    });

    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
    expect(checkMock).toHaveBeenCalledTimes(1);
  });

  it("the button stays disabled for the whole install", async () => {
    let finishInstall!: () => void;
    const downloadAndInstall = vi.fn(
      () => new Promise<void>((resolve) => (finishInstall = resolve)),
    );
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Install & Restart");
    });

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Installing…");
    });
    expect(button.disabled).toBe(true);

    button.click();
    finishInstall();
    await vi.waitFor(() => {
      expect(relaunchMock).toHaveBeenCalledTimes(1);
    });
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
    expect(checkMock).toHaveBeenCalledTimes(1);
  });

  it("a failed install re-arms the button for a retry that installs, not checks", async () => {
    const downloadAndInstall = vi
      .fn()
      .mockRejectedValueOnce(new Error("disk full"))
      .mockResolvedValueOnce(undefined);
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button, status } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Install & Restart");
    });
    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Retry install");
    });

    expect(status.textContent).toContain("disk full");
    expect(button.disabled).toBe(false);

    button.click();
    await vi.waitFor(() => {
      expect(downloadAndInstall).toHaveBeenCalledTimes(2);
    });
    expect(checkMock).toHaveBeenCalledTimes(1);
  });

  it("the button is usable again after a check that found nothing", async () => {
    checkMock.mockResolvedValue(null);
    const { button, status } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(status.textContent).toBe("You are up to date.");
    });

    expect(button.disabled).toBe(false);
    expect(button.textContent).toBe("Check for updates");

    button.click();
    await vi.waitFor(() => {
      expect(checkMock).toHaveBeenCalledTimes(2);
    });
  });

  it("a relaunch failure does not report the finished install as failed", async () => {
    const downloadAndInstall = vi.fn(async () => {});
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    relaunchMock.mockRejectedValue(new Error("process:allow-restart denied"));
    const { button, status } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Install & Restart");
    });
    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Check for updates");
    });

    expect(status.textContent).toContain("The update is installed");
    expect(status.textContent).not.toContain("Install failed");
    expect(button.disabled).toBe(false);

    button.click();
    await vi.waitFor(() => {
      expect(checkMock).toHaveBeenCalledTimes(2);
    });
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
  });

  it("a check that rejects with a non-Error still re-arms the button", async () => {
    checkMock.mockRejectedValue(null);
    const { button } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Check for updates");
    });

    expect(button.disabled).toBe(false);
  });

  it("the button is usable again after a check that failed", async () => {
    checkMock.mockRejectedValue(new Error("could not fetch a valid release json"));
    const { button, status } = renderUpdates();

    button.click();
    await vi.waitFor(() => {
      expect(status.textContent).toContain("not published yet");
    });

    expect(button.disabled).toBe(false);
    expect(button.textContent).toBe("Check for updates");

    button.click();
    await vi.waitFor(() => {
      expect(checkMock).toHaveBeenCalledTimes(2);
    });
  });
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

describe("renderGeneral — push-to-talk shortcut (spec 071)", () => {
  function withPlatform(value: string, run: () => void) {
    Object.defineProperty(navigator, "platform", { value, configurable: true });
    try {
      run();
    } finally {
      delete (navigator as { platform?: string }).platform;
    }
  }

  function capture(container: HTMLElement, init: KeyboardEventInit) {
    container.querySelector<HTMLButtonElement>("#btn-shortcut")!.click();
    document.dispatchEvent(new KeyboardEvent("keydown", init));
  }

  function hintOf(container: HTMLElement): HTMLElement {
    return container.querySelector<HTMLElement>("#shortcut-hint")!;
  }

  it("labels the button with the stored accelerator in the host platform's form", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    expect(container.querySelector("#btn-shortcut")!.textContent).toBe("Ctrl + Alt + V");
  });

  it("labels the button with Apple glyphs on a Mac navigator", () => {
    withPlatform("MacIntel", () => {
      const container = document.createElement("div");
      renderGeneral(container, buildSettings({ shortcut: "Super+Alt+KeyV" }));

      expect(container.querySelector("#btn-shortcut")!.textContent).toBe("⌥⌘V");
    });
  });

  it("requests the captured combination and never claims a restart is needed", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    capture(container, { key: "b", code: "KeyB", ctrlKey: true, altKey: true });

    expect(hintOf(container).textContent).toBe("Applying…");
    await vi.waitFor(() => {
      expect(emitMock).toHaveBeenCalledWith("shortcut-requested", { shortcut: "Ctrl+Alt+KeyB" });
    });
    expect(hintOf(container).textContent).not.toMatch(/restart/i);
  });

  it("writes nothing on the capture path — the widget persists only after it registers", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    capture(container, { key: "b", code: "KeyB", ctrlKey: true, altKey: true });

    await vi.waitFor(() => {
      expect(emitMock).toHaveBeenCalledWith("shortcut-requested", { shortcut: "Ctrl+Alt+KeyB" });
    });
    expect(apiMock.updateSettings).not.toHaveBeenCalled();
    expect(saveSettingsMock).not.toHaveBeenCalled();
    expect(emitMock).not.toHaveBeenCalledWith("settings-changed");
  });

  it("reports a failed request instead of leaving a success claim on screen", async () => {
    emitMock.mockRejectedValueOnce(new Error("bridge down"));

    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    capture(container, { key: "b", code: "KeyB", ctrlKey: true, altKey: true });

    await vi.waitFor(() => {
      expect(notifyErrorMock).toHaveBeenCalledWith("bridge down");
    });
    expect(hintOf(container).textContent).toBe("Could not apply the shortcut: bridge down");
    expect(container.querySelector("#btn-shortcut")!.textContent).toBe("Ctrl + Alt + V");
  });

  it("names the host platform's modifiers when none was held", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    capture(container, { key: "b", code: "KeyB" });

    expect(hintOf(container).textContent).toBe("Must include at least one modifier (Ctrl, Alt, Shift, Win)");
    expect(emitMock).not.toHaveBeenCalled();
  });

  it("replaces the hint with the outcome the widget reports", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });
    const handler = listenMock.mock.calls[0][1];

    handler({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: true,
        reason: null,
        persisted: true,
        stillActive: "Ctrl+Alt+KeyB",
      },
    });
    expect(hintOf(container).textContent).toBe("Ctrl + Alt + B is active now.");

    handler({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: false,
        reason: "already in use",
        persisted: null,
        stillActive: DEFAULT_SHORTCUT,
      },
    });
    expect(hintOf(container).textContent).toBe("Ctrl + Alt + B was not accepted: already in use");
  });

  it("puts the button label back to the combination that is still firing after a refusal", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    capture(container, { key: "b", code: "KeyB", ctrlKey: true, altKey: true });

    expect(container.querySelector("#btn-shortcut")!.textContent).toBe("Ctrl + Alt + B");
    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });

    listenMock.mock.calls[0][1]({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: false,
        reason: "already in use",
        persisted: null,
        stillActive: DEFAULT_SHORTCUT,
      },
    });

    expect(container.querySelector("#btn-shortcut")!.textContent).toBe("Ctrl + Alt + V");
  });

  it("says the shortcut is live but unsaved when the widget could not store it", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });

    listenMock.mock.calls[0][1]({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: true,
        reason: "backend down",
        persisted: false,
        stillActive: "Ctrl+Alt+KeyB",
      },
    });

    expect(hintOf(container).textContent).toBe(
      "Ctrl + Alt + B is active now, but could not be saved: backend down",
    );
  });

  it("refreshes the cached settings with the shortcut the widget stored", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });
    const handler = listenMock.mock.calls[0][1];

    handler({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: true,
        reason: null,
        persisted: true,
        stillActive: "Ctrl+Alt+KeyB",
      },
    });

    expect(cachePersistedShortcutMock).toHaveBeenCalledWith("Ctrl+Alt+KeyB");
  });

  it("leaves the cached settings alone when the widget could not store the shortcut", async () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });
    const handler = listenMock.mock.calls[0][1];

    handler({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: true,
        reason: "backend down",
        persisted: false,
        stillActive: "Ctrl+Alt+KeyB",
      },
    });
    handler({
      payload: {
        shortcut: "Ctrl+Alt+KeyB",
        ok: false,
        reason: "already in use",
        persisted: null,
        stillActive: DEFAULT_SHORTCUT,
      },
    });

    expect(cachePersistedShortcutMock).not.toHaveBeenCalled();
  });

  it("removes the very keydown listener it registered for the capture", () => {
    const addSpy = vi.spyOn(document, "addEventListener");
    const removeSpy = vi.spyOn(document, "removeEventListener");

    const container = document.createElement("div");
    const destroy = renderGeneral(container, buildSettings());
    container.querySelector<HTMLButtonElement>("#btn-shortcut")!.click();

    const added = addSpy.mock.calls.filter(([type]) => type === "keydown");
    expect(added).toHaveLength(1);

    destroy();

    const removed = removeSpy.mock.calls.filter(([type]) => type === "keydown");
    expect(removed).toHaveLength(1);
    expect(removed[0][1]).toBe(added[0][1]);
    expect(removed[0][2]).toBe(added[0][2]);

    addSpy.mockRestore();
    removeSpy.mockRestore();
  });

  it("a keydown already dispatched into the capture handler does nothing after destroy", async () => {
    const addSpy = vi.spyOn(document, "addEventListener");

    const container = document.createElement("div");
    const destroy = renderGeneral(container, buildSettings());
    container.querySelector<HTMLButtonElement>("#btn-shortcut")!.click();

    const registered = addSpy.mock.calls.find(([type]) => type === "keydown")!;
    const handler = registered[1] as (event: KeyboardEvent) => void;
    addSpy.mockRestore();

    destroy();
    handler(new KeyboardEvent("keydown", { key: "b", code: "KeyB", ctrlKey: true, altKey: true }));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(emitMock).not.toHaveBeenCalled();
    expect(container.querySelector("#btn-shortcut")!.textContent).toBe("Press keys...");
  });

  it("an abandoned capture leaves no keydown handler behind after destroy", async () => {
    const container = document.createElement("div");
    const destroy = renderGeneral(container, buildSettings());

    container.querySelector<HTMLButtonElement>("#btn-shortcut")!.click();
    destroy();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "f", code: "KeyF", ctrlKey: true }));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(emitMock).not.toHaveBeenCalled();
    expect(apiMock.updateSettings).not.toHaveBeenCalled();
    expect(saveSettingsMock).not.toHaveBeenCalled();
  });

  it("releases the shortcut-applied listener when the tab is destroyed", async () => {
    const container = document.createElement("div");
    const destroy = renderGeneral(container, buildSettings());

    await vi.waitFor(() => {
      expect(listenMock).toHaveBeenCalledWith("shortcut-applied", expect.any(Function));
    });
    destroy();

    expect(unlistenMock).toHaveBeenCalledTimes(1);
  });
});

describe("renderGeneral — meeting recording disclosure (spec 074, ADR 040)", () => {
  function group(container: HTMLElement): HTMLElement {
    return container.querySelector<HTMLElement>("#meeting-consent-group")!;
  }

  it("states that the user carries the consent obligation", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    const text = group(container).querySelector("#meeting-consent-responsibility")!.textContent!;

    expect(text).toMatch(/responsible/i);
    expect(text).toMatch(/consent/i);
  });

  it("states that Cloud mode sends the other participants' audio to the provider", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());

    const text = group(container).querySelector("#meeting-consent-cloud")!.textContent!;

    expect(text).toMatch(/cloud/i);
    expect(text).toMatch(/participants/i);
    expect(text).toMatch(/provider/i);
  });

  it("offers the acknowledgement while it has not been given", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings({ meeting_consent_acknowledged: false }));

    const button = group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!;

    expect(button.disabled).toBe(false);
    expect(group(container).querySelector("#meeting-consent-state")!.textContent).toMatch(
      /acknowledge/i,
    );
  });

  it("shows the already-acknowledged state instead of asking again", () => {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings({ meeting_consent_acknowledged: true }));

    const button = group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!;

    expect(button.disabled).toBe(true);
    expect(group(container).querySelector("#meeting-consent-state")!.textContent).toMatch(
      /already/i,
    );
  });

  it("persists the acknowledgement and repaints into the acknowledged state", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ meeting_consent_acknowledged: true }),
      warning: null,
    });

    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!.click();

    await vi.waitFor(() => {
      expect(saveSettingsMock).toHaveBeenCalledWith({ meeting_consent_acknowledged: true });
    });
    await vi.waitFor(() => {
      expect(
        group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!.disabled,
      ).toBe(true);
    });
  });

  it("re-enables the acknowledgement when it could not be saved", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));

    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    const button = group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!;
    button.click();

    await vi.waitFor(() => {
      expect(notifyErrorMock).toHaveBeenCalledWith("backend down");
    });
    expect(button.disabled).toBe(false);
  });

  it("both required statements survive into the acknowledged state", async () => {
    saveSettingsMock.mockResolvedValue({
      settings: buildSettings({ meeting_consent_acknowledged: true }),
      warning: null,
    });

    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    group(container).querySelector<HTMLButtonElement>("#btn-meeting-consent")!.click();

    await vi.waitFor(() => {
      expect(group(container).querySelector("#meeting-consent-cloud")).not.toBeNull();
    });
    expect(group(container).querySelector("#meeting-consent-responsibility")).not.toBeNull();
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

describe("renderGeneral — the microphone test", () => {
  function renderMicrophoneTest() {
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    return {
      button: container.querySelector<HTMLButtonElement>("#btn-test-mic")!,
      label: container.querySelector<HTMLElement>("#rec-label")!,
    };
  }

  it.each([
    ["a backend error", new Error("HTTP 500")],
    ["a connection that never reached it", new TypeError("Failed to fetch")],
  ])("does not claim the microphone is closed when the stop fails with %s", async (_name, failure) => {
    apiMock.audioStatus.mockResolvedValue({ is_recording: false, duration_seconds: 0, level_db: -60 });
    apiMock.audioStart.mockResolvedValue({ is_recording: true, duration_seconds: 0, level_db: -60 });
    apiMock.audioStop.mockRejectedValue(failure);
    const abort = vi.fn();
    levelStreamMock.mockImplementation(() => ({ abort }));
    const container = document.createElement("div");
    renderGeneral(container, buildSettings());
    const button = container.querySelector<HTMLButtonElement>("#btn-test-mic")!;
    const label = container.querySelector<HTMLElement>("#rec-label")!;
    const fill = container.querySelector<HTMLElement>("#level-fill")!;

    button.click();
    await vi.waitFor(() => {
      expect(button.textContent).toBe("Stop");
    });
    fill.style.width = "72%";

    button.click();
    await vi.waitFor(() => {
      expect(apiMock.audioStop).toHaveBeenCalledOnce();
    });

    expect(label.textContent).toBe("Stopping the microphone failed — it may still be open");
    expect(button.textContent).toBe("Stop");
    expect(abort).toHaveBeenCalled();
    expect(fill.style.width).toBe("0%");
    expect(consoleErrorMock).toHaveBeenCalledWith(failure);

    const onError = levelStreamMock.mock.calls[0][2] as (error: string) => void;
    onError("the backend did not answer /audio/level-stream within 15 seconds");
    expect(label.textContent).toBe("Stopping the microphone failed — it may still be open");
  });

  it("still reports an ordinary start failure as a failure", async () => {
    apiMock.audioStatus.mockResolvedValue({ is_recording: false, duration_seconds: 0, level_db: -60 });
    apiMock.audioStart.mockRejectedValue(new Error("connection refused"));
    const { button, label } = renderMicrophoneTest();

    button.click();
    await vi.waitFor(() => {
      expect(label.textContent).toBe("Failed to start");
    });

    expect(button.textContent).toBe("Record");
  });

  it("puts an expired level-stream handshake into the label and leaves the recording alone", async () => {
    apiMock.audioStatus.mockResolvedValue({ is_recording: false, duration_seconds: 0, level_db: -60 });
    apiMock.audioStart.mockResolvedValue({ is_recording: true, duration_seconds: 0, level_db: -60 });
    const { button, label } = renderMicrophoneTest();

    button.click();
    await vi.waitFor(() => {
      expect(levelStreamMock).toHaveBeenCalledOnce();
    });

    const onError = levelStreamMock.mock.calls[0][2] as (error: string) => void;
    onError(new TimedOutError(REQUEST_TIMEOUT_MS, "/audio/level-stream").message);

    expect(label.textContent).toContain("the level meter stopped");
    expect(label.textContent).toContain("did not answer /audio/level-stream");
    expect(button.textContent).toBe("Stop");
    expect(apiMock.audioStop).not.toHaveBeenCalled();
  });
});
