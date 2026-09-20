// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LocalSTTStatus, UserSettings } from "../../api";

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

let cleanups: Array<() => void> = [];

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
  for (const cleanup of cleanups) cleanup();
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

  it("stops polling once the cleanup it returns is called", async () => {
    vi.useFakeTimers();
    try {
      await render(buildStatus());
      await vi.advanceTimersByTimeAsync(0);
      const afterFirstRender = apiMock.sttLocalStatus.mock.calls.length;
      expect(afterFirstRender).toBeGreaterThan(0);

      for (const cleanup of cleanups) cleanup();
      cleanups = [];
      await vi.advanceTimersByTimeAsync(9000);

      expect(apiMock.sttLocalStatus.mock.calls.length).toBe(afterFirstRender);
    } finally {
      vi.useRealTimers();
    }
  });
});
