// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CloudKeyStatus, LocalSTTStatus, UserSettings } from "../../api";
import type { TabLifecycle } from "../settings";

const apiMock = {
  sttLocalStatus: vi.fn(),
  sttLocalPrewarm: vi.fn(),
  setSttMode: vi.fn(),
  updateSettings: vi.fn(),
};

vi.mock("../../api", () => ({
  api: apiMock,
}));

const cloudStatusMock = vi.fn((): CloudKeyStatus | null => ({ gemini_key_set: true, groq_key_set: true }));
vi.mock("../settings", () => ({
  loadSettings: vi.fn(async () => {}),
  getCloudKeyStatus: () => cloudStatusMock(),
}));

const notifyErrorMock = vi.fn();
vi.mock("../../notify", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../notify")>();
  return { ...actual, notifyError: notifyErrorMock };
});

const BLANK_SHAPES = [
  ["an empty string", ""],
  ["spaces alone", "   "],
  ["a newline alone", "\n"],
  ["a byte-order mark", "\ufeff"],
  ['what `str(KeyError(""))` produces', "''"],
] as const;

function buildSettings(overrides: Partial<UserSettings> = {}): UserSettings {
  return {
    language: "uk",
    shortcut: "Ctrl+Alt+KeyV",
    output_dir: "C:/fake",
    stt_mode: "local",
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
    display_name: "",
    ...overrides,
  };
}

function buildStatus(overrides: Partial<LocalSTTStatus> = {}): LocalSTTStatus {
  return {
    available: true,
    package_installed: true,
    model_downloaded: true,
    model_bytes: 1_624_555_275,
    model_loaded: false,
    model_name: "large-v3-turbo",
    model_ram_mb: null,
    gpu_available: false,
    gpu_name: null,
    gpu_vendor: "none",
    device: "cpu",
    compute_type: "int8",
    last_error: null,
    ...overrides,
  };
}

let cleanups: TabLifecycle[] = [];

async function render(
  status: LocalSTTStatus,
  settings: Partial<UserSettings> = {},
): Promise<HTMLElement> {
  apiMock.sttLocalStatus.mockResolvedValue(status);
  const { renderDictationMode } = await import("./dictation-mode");
  const container = document.createElement("div");
  cleanups.push(renderDictationMode(container, buildSettings(settings)));
  return container;
}

function row(container: HTMLElement, mode: "cloud" | "local"): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>(`#mode-${mode}`)!;
}

function hint(container: HTMLElement, mode: "cloud" | "local"): HTMLElement {
  return row(container, mode).querySelector<HTMLElement>(".mode-row-hint")!;
}

function action(container: HTMLElement): string {
  return row(container, "local").querySelector<HTMLElement>(".mode-row-action")!.textContent!.trim();
}

function failureShown(container: HTMLElement): boolean {
  return action(container) === "Try again" && hint(container, "local").classList.contains("mode-row-alert");
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  apiMock.setSttMode.mockResolvedValue({});
  apiMock.sttLocalPrewarm.mockResolvedValue({ started: true });
});

afterEach(() => {
  for (const cleanup of cleanups) cleanup.destroy();
  cleanups = [];
});

describe("renderDictationMode — a failed local load the user can read", () => {
  it.each(BLANK_SHAPES)(
    "draws %s as one readable toast and a failed row",
    async (_label, raw) => {
      const container = await render(buildStatus({ last_error: raw }));

      await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
      expect(notifyErrorMock.mock.calls[0][0]).toMatch(/[\p{L}\p{N}]/u);
      expect(failureShown(container)).toBe(true);
      expect(hint(container, "local").textContent).toBe(notifyErrorMock.mock.calls[0][0]);
    },
  );

  it.each(BLANK_SHAPES)(
    "never draws a ready row over %s reported beside a loaded model",
    async (_label, raw) => {
      const container = await render(buildStatus({ last_error: raw, model_loaded: true }));

      await vi.waitFor(() => expect(failureShown(container)).toBe(true));
      expect(hint(container, "local").textContent).toMatch(/[\p{L}\p{N}]/u);
    },
  );

  it("passes a message that already reads as text through unchanged", async () => {
    const container = await render(buildStatus({ last_error: "The model file is missing." }));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
    expect(notifyErrorMock.mock.calls[0][0]).toBe("The model file is missing.");
    expect(hint(container, "local").textContent).toBe("The model file is missing.");
  });

  it("raises no toast and draws a ready row when nothing failed", async () => {
    const container = await render(buildStatus({ model_loaded: true }));

    await vi.waitFor(() => expect(action(container)).toBe("Ready"));
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });

  it("polls while it is mounted and stops once the cleanup it returns is called", async () => {
    vi.useFakeTimers();
    try {
      await render(buildStatus());
      await vi.advanceTimersByTimeAsync(0);
      const afterFirstRender = apiMock.sttLocalStatus.mock.calls.length;
      expect(afterFirstRender).toBeGreaterThan(0);

      await vi.advanceTimersByTimeAsync(3000);
      const whilePolling = apiMock.sttLocalStatus.mock.calls.length;
      expect(whilePolling).toBeGreaterThan(afterFirstRender);

      for (const cleanup of cleanups) cleanup.destroy();
      cleanups = [];
      await vi.advanceTimersByTimeAsync(9000);

      expect(apiMock.sttLocalStatus.mock.calls.length).toBe(whilePolling);
    } finally {
      vi.useRealTimers();
    }
  });

  it("raises a second toast when a different failure normalizes to the same sentence", async () => {
    vi.useFakeTimers();
    try {
      apiMock.sttLocalStatus
        .mockResolvedValueOnce(buildStatus({ last_error: "" }))
        .mockResolvedValue(buildStatus({ last_error: "\n" }));
      const { renderDictationMode } = await import("./dictation-mode");
      const container = document.createElement("div");
      cleanups.push(renderDictationMode(container, buildSettings()));

      await vi.advanceTimersByTimeAsync(0);
      expect(notifyErrorMock).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(3000);
      expect(notifyErrorMock).toHaveBeenCalledTimes(2);
      expect(notifyErrorMock.mock.calls[1][0]).toBe(notifyErrorMock.mock.calls[0][0]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("raises the toast again when the tab is reopened while the failure is still there", async () => {
    const container = await render(buildStatus({ last_error: "" }));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
    expect(failureShown(container)).toBe(true);

    for (const cleanup of cleanups) cleanup.destroy();
    cleanups = [];

    const { renderDictationMode } = await import("./dictation-mode");
    const reopened = document.createElement("div");
    cleanups.push(renderDictationMode(reopened, buildSettings()));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(2));
  });

  it("ignores a status poll that was already in flight when the tab was torn down", async () => {
    let resolveFirst: (s: LocalSTTStatus) => void = () => {};
    apiMock.sttLocalStatus.mockImplementationOnce(
      () => new Promise<LocalSTTStatus>((resolve) => (resolveFirst = resolve)),
    );
    const { renderDictationMode } = await import("./dictation-mode");
    renderDictationMode(document.createElement("div"), buildSettings()).destroy();

    resolveFirst(buildStatus({ last_error: "boom" }));
    await new Promise((done) => setTimeout(done, 0));

    expect(notifyErrorMock).not.toHaveBeenCalled();

    apiMock.sttLocalStatus.mockResolvedValue(buildStatus({ last_error: "boom" }));
    const reopened = document.createElement("div");
    cleanups.push(renderDictationMode(reopened, buildSettings()));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
  });

  it("draws no error and raises no toast when the field is absent from the response", async () => {
    const status = buildStatus({ model_loaded: true });
    delete (status as { last_error?: string | null }).last_error;
    const container = await render(status);

    await vi.waitFor(() => expect(action(container)).toBe("Ready"));
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });
});

describe("renderDictationMode after the Settings window is dismissed", () => {
  it("issues no further status read once the window is dismissed", async () => {
    vi.useFakeTimers();
    try {
      await render(buildStatus());
      await vi.advanceTimersByTimeAsync(0);
      const [tab] = cleanups;
      tab.releaseResources!();
      const whileHidden = apiMock.sttLocalStatus.mock.calls.length;
      expect(whileHidden).toBeGreaterThan(0);

      await vi.advanceTimersByTimeAsync(30_000);

      expect(apiMock.sttLocalStatus.mock.calls.length).toBe(whileHidden);
    } finally {
      vi.useRealTimers();
    }
  });

  it("reads once on the way back before any tick, then keeps the 3 s rhythm", async () => {
    vi.useFakeTimers();
    try {
      await render(buildStatus());
      await vi.advanceTimersByTimeAsync(0);
      const [tab] = cleanups;
      tab.releaseResources!();
      const whileHidden = apiMock.sttLocalStatus.mock.calls.length;

      tab.resumeResources!();

      expect(
        apiMock.sttLocalStatus.mock.calls.length,
        "a returning user reads a row one request old, not one interval old",
      ).toBe(whileHidden + 1);

      await vi.advanceTimersByTimeAsync(3000);
      expect(apiMock.sttLocalStatus.mock.calls.length).toBe(whileHidden + 2);

      await vi.advanceTimersByTimeAsync(3000);
      expect(
        apiMock.sttLocalStatus.mock.calls.length,
        "a resume that started a second interval would read twice per tick",
      ).toBe(whileHidden + 3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("starts no second interval when a resume lands on a poll already running", async () => {
    vi.useFakeTimers();
    try {
      await render(buildStatus());
      await vi.advanceTimersByTimeAsync(0);
      const [tab] = cleanups;

      tab.resumeResources!();
      const afterResume = apiMock.sttLocalStatus.mock.calls.length;

      await vi.advanceTimersByTimeAsync(3000);
      expect(
        apiMock.sttLocalStatus.mock.calls.length,
        "the tab can be mounted with the window already up, and a resume that overwrites " +
          "the live handle reads twice per tick",
      ).toBe(afterResume + 1);

      tab.releaseResources!();
      await vi.advanceTimersByTimeAsync(30_000);

      expect(
        apiMock.sttLocalStatus.mock.calls.length,
        "and leaves an interval the release can no longer reach",
      ).toBe(afterResume + 1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("reads nothing at all when it mounts into a dismissed window", async () => {
    vi.useFakeTimers();
    try {
      apiMock.sttLocalStatus.mockResolvedValue(buildStatus());
      const { renderDictationMode } = await import("./dictation-mode");
      const container = document.createElement("div");
      const tab = renderDictationMode(container, buildSettings(), true);
      cleanups.push(tab);
      await vi.advanceTimersByTimeAsync(0);

      expect(
        apiMock.sttLocalStatus,
        "the mount's own read is away before any release can run, so a tab mounted into " +
          "a window nobody can see still costs a request",
      ).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(30_000);
      expect(apiMock.sttLocalStatus).not.toHaveBeenCalled();

      tab.resumeResources!();
      expect(
        apiMock.sttLocalStatus,
        "and the show is what pays for it, once",
      ).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("raises no second toast for a failure the user has already been shown", async () => {
    vi.useFakeTimers();
    try {
      const container = await render(buildStatus({ last_error: "boom" }));
      await vi.advanceTimersByTimeAsync(0);
      expect(notifyErrorMock).toHaveBeenCalledTimes(1);
      const [tab] = cleanups;

      for (let reopen = 0; reopen < 3; reopen += 1) {
        tab.releaseResources!();
        tab.resumeResources!();
        await vi.advanceTimersByTimeAsync(0);
      }

      expect(
        notifyErrorMock,
        "closing and re-opening Settings against a local engine that is still broken " +
          "must not raise the same toast again on every re-open",
      ).toHaveBeenCalledTimes(1);
      expect(failureShown(container), "while the row still says the engine is broken").toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("renderDictationMode — the two rows", () => {
  it.each([
    ["an engine this computer cannot run", buildStatus({ available: false }), "cloud", "Not available on this computer", ""],
    [
      "a model not yet downloaded",
      buildStatus({ model_downloaded: false }),
      "cloud",
      "Not installed · 1.5 GB",
      "Install",
    ],
    [
      "a missing engine package",
      buildStatus({ package_installed: false, model_downloaded: false, model_bytes: 487_601_967 }),
      "cloud",
      "Not installed · 465 MB",
      "Install",
    ],
    ["a model of no known size", buildStatus({ model_downloaded: false, model_bytes: null }), "cloud", "Not installed", "Install"],
    ["an installed engine while Cloud is on", buildStatus(), "cloud", "Installed · runs on this computer", "Ready"],
    ["a loaded engine", buildStatus({ model_loaded: true }), "local", "Installed · runs on this computer", "Ready"],
    ["an engine still loading", buildStatus(), "local", "Starting…", ""],
    ["an engine still downloading", buildStatus({ model_downloaded: false }), "local", "Starting…", ""],
    ["a failed engine", buildStatus({ last_error: "CUDA out of memory" }), "local", "CUDA out of memory", "Try again"],
    [
      "a failure left over from Local while Cloud is on",
      buildStatus({ last_error: "CUDA out of memory" }),
      "cloud",
      "Installed · runs on this computer",
      "Ready",
    ],
    [
      "a refusal on a computer with no engine",
      buildStatus({ available: false, package_installed: false, last_error: "No engine here." }),
      "local",
      "Not available on this computer",
      "",
    ],
  ] as const)("draws %s", async (_label, status, mode, expectedHint, expectedAction) => {
    const container = await render(status, { stt_mode: mode });

    await vi.waitFor(() => expect(hint(container, "local").textContent).toBe(expectedHint));
    expect(action(container)).toBe(expectedAction);
    expect(row(container, "local").getAttribute("aria-checked")).toBe(String(mode === "local"));
    expect(row(container, "cloud").getAttribute("aria-checked")).toBe(String(mode === "cloud"));
  });

  it("dims a model not yet downloaded without telling a screen reader it cannot be used", async () => {
    const container = await render(buildStatus({ model_downloaded: false }), { stt_mode: "cloud" });
    await vi.waitFor(() => expect(action(container)).toBe("Install"));

    expect(row(container, "local").classList.contains("mode-row--locked")).toBe(true);
    expect(row(container, "local").hasAttribute("aria-disabled")).toBe(false);
  });

  it("raises no second toast for the same failure after a trip through Cloud", async () => {
    const container = await render(buildStatus({ last_error: "CUDA out of memory" }), { stt_mode: "local" });
    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledOnce());

    row(container, "cloud").click();
    await vi.waitFor(() => expect(apiMock.setSttMode).toHaveBeenCalledWith("cloud"));
    await vi.waitFor(() => expect(apiMock.sttLocalStatus.mock.calls.length).toBeGreaterThan(1));
    await new Promise((done) => setTimeout(done, 0));
    expect(action(container)).toBe("Ready");
    row(container, "local").click();
    const readsBeforeLocal = apiMock.sttLocalStatus.mock.calls.length;
    await vi.waitFor(() => expect(apiMock.sttLocalStatus.mock.calls.length).toBeGreaterThan(readsBeforeLocal));
    await new Promise((done) => setTimeout(done, 0));

    expect(action(container)).toBe("Try again");
    expect(notifyErrorMock).toHaveBeenCalledOnce();
  });

  it("dims the row of an engine this computer cannot run and ignores a click on it", async () => {
    const container = await render(buildStatus({ available: false }), { stt_mode: "cloud" });
    await vi.waitFor(() => expect(hint(container, "local").textContent).toBe("Not available on this computer"));

    row(container, "local").click();
    await new Promise((done) => setTimeout(done, 0));

    expect(row(container, "local").classList.contains("mode-row--locked")).toBe(true);
    expect(row(container, "local").getAttribute("aria-disabled")).toBe("true");
    expect(apiMock.setSttMode).not.toHaveBeenCalled();
    expect(apiMock.sttLocalPrewarm).not.toHaveBeenCalled();
    expect(row(container, "cloud").getAttribute("aria-checked")).toBe("true");
  });

  it("ignores a click on the Local row before the first status read answers", async () => {
    apiMock.sttLocalStatus.mockReturnValue(new Promise(() => {}));
    const { renderDictationMode } = await import("./dictation-mode");
    const container = document.createElement("div");
    cleanups.push(renderDictationMode(container, buildSettings({ stt_mode: "cloud" })));

    row(container, "local").click();
    await new Promise((done) => setTimeout(done, 0));

    expect(apiMock.setSttMode).not.toHaveBeenCalled();
  });

  it("switches to Local and shows Starting… when Install is clicked", async () => {
    const container = await render(buildStatus({ model_downloaded: false }), { stt_mode: "cloud" });
    await vi.waitFor(() => expect(action(container)).toBe("Install"));

    row(container, "local").click();

    expect(row(container, "local").getAttribute("aria-checked")).toBe("true");
    expect(hint(container, "local").textContent).toBe("Starting…");
    await vi.waitFor(() => expect(apiMock.setSttMode).toHaveBeenCalledWith("local"));
  });

  it("switches back to Cloud when its row is picked", async () => {
    const container = await render(buildStatus({ model_loaded: true }), { stt_mode: "local" });
    await vi.waitFor(() => expect(action(container)).toBe("Ready"));

    row(container, "cloud").click();

    expect(row(container, "cloud").getAttribute("aria-checked")).toBe("true");
    await vi.waitFor(() => expect(apiMock.setSttMode).toHaveBeenCalledWith("cloud"));
  });

  it("puts the row back and says why when the mode could not be saved", async () => {
    apiMock.setSttMode.mockRejectedValue(new Error("Backend said no"));
    const container = await render(buildStatus(), { stt_mode: "cloud" });
    await vi.waitFor(() => expect(action(container)).toBe("Ready"));

    row(container, "local").click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("Backend said no"));
    expect(row(container, "cloud").getAttribute("aria-checked")).toBe("true");
  });

  it("retries the engine load on Try again", async () => {
    const container = await render(buildStatus({ last_error: "CUDA out of memory" }), { stt_mode: "local" });
    await vi.waitFor(() => expect(action(container)).toBe("Try again"));

    row(container, "local").click();

    await vi.waitFor(() => expect(apiMock.sttLocalPrewarm).toHaveBeenCalledOnce());
    expect(apiMock.setSttMode).not.toHaveBeenCalled();
  });

  it("offers Try again that only reads again when the status read itself failed", async () => {
    apiMock.sttLocalStatus.mockRejectedValue(new Error("offline"));
    const { renderDictationMode } = await import("./dictation-mode");
    const container = document.createElement("div");
    cleanups.push(renderDictationMode(container, buildSettings({ stt_mode: "local" })));
    await vi.waitFor(() => expect(hint(container, "local").textContent).toBe("Couldn't load this"));
    const reads = apiMock.sttLocalStatus.mock.calls.length;

    apiMock.sttLocalStatus.mockResolvedValue(buildStatus({ model_loaded: true }));
    row(container, "local").click();

    await vi.waitFor(() => expect(action(container)).toBe("Ready"));
    expect(apiMock.sttLocalStatus.mock.calls.length).toBeGreaterThan(reads);
    expect(apiMock.sttLocalPrewarm).not.toHaveBeenCalled();
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });

  it.each([
    ["auto", { groq_key_set: true, gemini_key_set: true }, false],
    ["auto", { groq_key_set: true, gemini_key_set: false }, true],
    ["auto", { groq_key_set: false, gemini_key_set: true }, true],
    ["groq", { groq_key_set: true, gemini_key_set: false }, false],
    ["groq", { groq_key_set: false, gemini_key_set: true }, true],
    ["gemini", { groq_key_set: false, gemini_key_set: true }, false],
    ["gemini", { groq_key_set: true, gemini_key_set: false }, true],
  ] as const)("with recordings going to %s and keys %o, asks for a key: %s", async (engine, keys, missing) => {
    cloudStatusMock.mockReturnValueOnce(keys);
    const container = await render(buildStatus(), { stt_engine: engine });

    expect(hint(container, "cloud").textContent).toBe(
      missing ? "Add an API key in Settings" : "Your API keys · fastest and most accurate",
    );
    expect(hint(container, "cloud").classList.contains("mode-row-alert")).toBe(missing);
  });

  it("does not ask for a key it could not check", async () => {
    cloudStatusMock.mockReturnValueOnce(null);
    const container = await render(buildStatus());

    expect(hint(container, "cloud").textContent).toBe("Your API keys · fastest and most accurate");
  });
});
