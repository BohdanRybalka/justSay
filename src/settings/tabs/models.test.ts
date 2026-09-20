// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LocalSTTStatus, UserSettings } from "../../api";
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

vi.mock("../settings", () => ({
  loadSettings: vi.fn(),
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
    ...overrides,
  };
}

function buildStatus(overrides: Partial<LocalSTTStatus> = {}): LocalSTTStatus {
  return {
    package_installed: true,
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

async function render(status: LocalSTTStatus): Promise<HTMLElement> {
  apiMock.sttLocalStatus.mockResolvedValue(status);
  const { renderModels } = await import("./models");
  const container = document.createElement("div");
  cleanups.push(renderModels(container, buildSettings()));
  return container;
}

function badgeClass(container: HTMLElement): string {
  return container.querySelector<HTMLElement>("#stt-local-indicator")!.className;
}

function badgeTitle(container: HTMLElement): string {
  return container.querySelector<HTMLElement>("#stt-local-indicator")!.title;
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
});

afterEach(() => {
  for (const cleanup of cleanups) cleanup.destroy();
  cleanups = [];
});

describe("renderModels — a failed local load the user can read", () => {
  it.each(BLANK_SHAPES)(
    "draws %s as one readable toast and an error badge",
    async (_label, raw) => {
      const container = await render(buildStatus({ last_error: raw }));

      await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
      expect(notifyErrorMock.mock.calls[0][0]).toMatch(/[\p{L}\p{N}]/u);
      expect(badgeClass(container)).toContain("status-indicator-badge--error");
      expect(badgeClass(container)).not.toContain("status-indicator-badge--ready");
    },
  );

  it.each(BLANK_SHAPES)(
    "never draws a healthy badge over %s reported beside a loaded model",
    async (_label, raw) => {
      const container = await render(buildStatus({ last_error: raw, model_loaded: true }));

      await vi.waitFor(() =>
        expect(badgeClass(container)).toContain("status-indicator-badge--error"),
      );
      expect(badgeClass(container)).not.toContain("status-indicator-badge--ready");
      expect(badgeTitle(container)).toMatch(/[\p{L}\p{N}]/u);
    },
  );

  it("puts the same readable sentence in the tooltip and the aria-label", async () => {
    const container = await render(buildStatus({ last_error: "''" }));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
    const badge = container.querySelector<HTMLElement>("#stt-local-indicator")!;
    const shown = notifyErrorMock.mock.calls[0][0];
    expect(shown).toMatch(/[\p{L}\p{N}]/u);
    expect(badge.title).toBe(shown);
    expect(badge.getAttribute("aria-label")).toContain(shown);
  });

  it("passes a message that already reads as text through unchanged", async () => {
    const container = await render(buildStatus({ last_error: "The model file is missing." }));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
    expect(notifyErrorMock.mock.calls[0][0]).toBe("The model file is missing.");
    expect(badgeClass(container)).toContain("status-indicator-badge--error");
  });

  it("raises no toast and draws a ready badge when nothing failed", async () => {
    const container = await render(buildStatus({ model_loaded: true }));

    await vi.waitFor(() => expect(badgeClass(container)).toContain("status-indicator-badge--ready"));
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
      const { renderModels } = await import("./models");
      const container = document.createElement("div");
      cleanups.push(renderModels(container, buildSettings()));

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
    expect(badgeClass(container)).toContain("status-indicator-badge--error");

    for (const cleanup of cleanups) cleanup.destroy();
    cleanups = [];

    const { renderModels } = await import("./models");
    const reopened = document.createElement("div");
    cleanups.push(renderModels(reopened, buildSettings()));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(2));
  });

  it("ignores a status poll that was already in flight when the tab was torn down", async () => {
    let resolveFirst: (s: LocalSTTStatus) => void = () => {};
    apiMock.sttLocalStatus.mockImplementationOnce(
      () => new Promise<LocalSTTStatus>((resolve) => (resolveFirst = resolve)),
    );
    const { renderModels } = await import("./models");
    renderModels(document.createElement("div"), buildSettings()).destroy();

    resolveFirst(buildStatus({ last_error: "boom" }));
    await new Promise((done) => setTimeout(done, 0));

    expect(notifyErrorMock).not.toHaveBeenCalled();

    apiMock.sttLocalStatus.mockResolvedValue(buildStatus({ last_error: "boom" }));
    const reopened = document.createElement("div");
    cleanups.push(renderModels(reopened, buildSettings()));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledTimes(1));
  });

  it("keeps the caption free of `undefined` when the response omits the model fields", async () => {
    const status = buildStatus({ model_loaded: true });
    delete (status as { model_name?: string }).model_name;
    delete (status as { device?: string }).device;
    const container = await render(status);

    await vi.waitFor(() => expect(badgeClass(container)).toContain("status-indicator-badge--ready"));
    const caption = container.querySelector<HTMLElement>("#stt-local-caption")!;
    expect(caption.textContent).toBeTruthy();
    expect(caption.textContent).not.toContain("undefined");
  });

  it("draws no error and raises no toast when the field is absent from the response", async () => {
    const status = buildStatus({ model_loaded: true });
    delete (status as { last_error?: string | null }).last_error;
    const container = await render(status);

    await vi.waitFor(() => expect(badgeClass(container)).toContain("status-indicator-badge--ready"));
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });
});

describe("renderModels after the Settings window is dismissed", () => {
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
        "a returning user reads a badge one request old, not one interval old",
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
});
