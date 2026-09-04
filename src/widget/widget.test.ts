// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RecordingStatus } from "../api";

const apiMock = {
  health: vi.fn(async () => ({ status: "ok", version: "0", stt_mode: "cloud", llm_mode: "cloud" })),
  getSettings: vi.fn(async () => {
    throw new Error("settings not needed here");
  }),
  updateSettings: vi.fn(),
  getMeetingStatus: vi.fn(async () => ({
    is_recording: false,
    duration_seconds: 0,
    level_db: -60,
    system_endpoint: null,
    system_level_db: -60,
  })),
  startMeetingRecording: vi.fn(),
  stopMeetingRecording: vi.fn(),
  audioStart: vi.fn(),
  audioStatus: vi.fn(),
  dictate: vi.fn(),
};

const notifyErrorMock = vi.fn();

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: apiMock };
});

vi.mock("../notify", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../notify")>();
  return { ...actual, notifyError: notifyErrorMock };
});

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => {}) }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async () => () => {}),
  emit: vi.fn(async () => {}),
}));
vi.mock("@tauri-apps/plugin-global-shortcut", () => ({
  register: vi.fn(async () => {}),
  unregister: vi.fn(async () => {}),
  isRegistered: vi.fn(async () => false),
}));

/** Built from the module registry `vi.resetModules()` has just installed, which
 *  is the one `widget.ts` will import: a `TimedOutError` from any other
 *  instance of the module fails `instanceof` and the test would then pass
 *  through the generic failure branch rather than the one under test. */
async function timedOut(path: string): Promise<Error> {
  const { REQUEST_TIMEOUT_MS } = await import("../api");
  const { TimedOutError } = await import("../timeout");
  return new TimedOutError(REQUEST_TIMEOUT_MS, path);
}

function recordingStatus(overrides: Partial<RecordingStatus> = {}): RecordingStatus {
  return { is_recording: false, duration_seconds: 0, level_db: -60, ...overrides };
}

function widgetBody(): string {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  return html.slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"));
}

async function loadWidget() {
  document.body.innerHTML = widgetBody().replace(/<script[\s\S]*?<\/script>/g, "");
  await import("./widget");
  await vi.waitFor(() => expect(apiMock.health).toHaveBeenCalled());
}

function root(): HTMLElement {
  return document.getElementById("widget")!;
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  apiMock.health.mockResolvedValue({
    status: "ok",
    version: "0",
    stt_mode: "cloud",
    llm_mode: "cloud",
  });
  apiMock.getSettings.mockRejectedValue(new Error("settings not needed here"));
  apiMock.getMeetingStatus.mockResolvedValue({
    is_recording: false,
    duration_seconds: 0,
    level_db: -60,
    system_endpoint: null,
    system_level_db: -60,
  });
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("a dictation start that runs out of its budget", () => {
  it("stays in recording, showing the backend's elapsed time, when the microphone really is open", async () => {
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ is_recording: true, duration_seconds: 7.5 }),
    );
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(apiMock.audioStatus).toHaveBeenCalled());
    await vi.waitFor(() =>
      expect(document.getElementById("widget-duration")!.textContent).not.toBe("0.0s"),
    );

    expect(root().className).toBe("widget recording");
    expect(document.getElementById("widget-text")!.textContent).toBe("Recording");
    expect(document.getElementById("widget-duration")!.textContent).toMatch(/^7\.\ds$/);
    expect(notifyErrorMock).not.toHaveBeenCalledWith("Couldn't start recording — try again.");
  });

  it("shows the start as failed when the backend reports no recording", async () => {
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(recordingStatus({ is_recording: false }));
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));

    expect(apiMock.audioStatus).toHaveBeenCalledOnce();
    expect(document.getElementById("widget-text")!.textContent).toBe("Start failed");
    expect(notifyErrorMock).toHaveBeenCalledWith("Couldn't start recording — try again.");
  });

  it("shows the start as failed when the status read cannot answer either", async () => {
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockRejectedValue(await timedOut("/audio/status"));
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));

    expect(apiMock.audioStatus).toHaveBeenCalledOnce();
    expect(document.getElementById("widget-text")!.textContent).toBe("Start failed");
  });

  it("reads no status at all when the start fails for an ordinary reason", async () => {
    apiMock.audioStart.mockRejectedValue(new Error("connection refused"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ is_recording: true, duration_seconds: 7.5 }),
    );
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));

    expect(apiMock.audioStatus).not.toHaveBeenCalled();
  });
});
