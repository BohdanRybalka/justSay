// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CONNECTION_POLL_MS } from "./settings-retry";

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
  audioStop: vi.fn(),
  audioStatus: vi.fn(),
  dictate: vi.fn(),
};

const listeners = new Map<string, (event: unknown) => unknown>();

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
  listen: vi.fn(async (event: string, handler: (payload: unknown) => unknown) => {
    listeners.set(event, handler);
    return () => {};
  }),
  emit: vi.fn(async () => {}),
}));
vi.mock("@tauri-apps/plugin-global-shortcut", () => ({
  register: vi.fn(async () => {}),
  unregister: vi.fn(async () => {}),
  isRegistered: vi.fn(async () => false),
}));

function widgetBody(): string {
  const html = readFileSync(resolve(__dirname, "../../widget.html"), "utf-8");
  return html.slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"));
}

async function loadWidget() {
  document.body.innerHTML = widgetBody().replace(/<script[\s\S]*?<\/script>/g, "");
  await import("./widget");
  await vi.waitFor(() => expect(apiMock.health).toHaveBeenCalled());
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetModules();
  vi.clearAllMocks();
  listeners.clear();
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

afterEach(() => {
  vi.useRealTimers();
});

describe("the widget's own timers", () => {
  it("advances only its own connection poll, whatever ran before it", async () => {
    await loadWidget();
    const before = apiMock.health.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);

    expect(apiMock.health.mock.calls.length - before).toBe(3);
  });

  it("never runs two health probes at once, so the connection state is written in order", async () => {
    await loadWidget();
    let release = () => {};
    apiMock.health.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = () => resolve({ status: "ok", version: "0", stt_mode: "cloud", llm_mode: "cloud" });
        }),
    );
    const before = apiMock.health.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 4);

    expect(apiMock.health.mock.calls.length - before).toBe(1);
    release();
  });
});

describe("a start the backend refuses", () => {
  it("names the layer that refused it rather than telling the user to try again", async () => {
    const { ApiAuthError } = await import("../api");
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(
      new ApiAuthError("Missing or invalid API token", { kind: "invoke-timeout" }),
    );

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(apiMock.audioStart).toHaveBeenCalledOnce();
    });

    expect(document.getElementById("widget-text")!.textContent).toBe("Auth failed");
    expect(notifyErrorMock).toHaveBeenCalledWith(
      "JustSay could not authenticate to its own backend — restart the app.",
    );
  });
});
