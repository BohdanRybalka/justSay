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
  it.each([
    ["a refusal whose body contains the word 'missing'", new Error("Missing or invalid API token")],
    ["a 409 that means the recorder is already held", new Error("Already recording")],
  ])("does not describe %s as a dictation or as a missing cloud key", async (_name, failure) => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(failure);

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(apiMock.audioStart).toHaveBeenCalledOnce();
    });

    expect(document.getElementById("widget-text")!.textContent).toBe("Start failed");
    expect(notifyErrorMock).toHaveBeenCalledWith("Couldn't start recording — try again.");
  });

  it("reverts to idle on a failure that was observed, because nothing is left running", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(new Error("connection refused"));

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.getElementById("widget-text")!.textContent).toBe("Start failed");
    });

    await vi.advanceTimersByTimeAsync(3000);

    expect(document.getElementById("widget-text")!.textContent).toBe("JustSay");
  });
});

describe("a start the backend never answered", () => {
  it("says the microphone may still be open and stays there", async () => {
    const { TimedOutError } = await import("../timeout");
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(new TimedOutError(60_000, "/audio/start"));

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.getElementById("widget-text")!.textContent).toBe("No answer");
    });

    expect(notifyErrorMock).toHaveBeenCalledWith(
      "The backend never answered — the microphone may still be open.",
    );

    await vi.advanceTimersByTimeAsync(60_000);

    expect(document.getElementById("widget-text")!.textContent).toBe("No answer");
  });

  it("keeps the same refusal to revert after an abandoned dictation", async () => {
    const { TimedOutError } = await import("../timeout");
    await loadWidget();
    apiMock.audioStart.mockResolvedValue({
      is_recording: true,
      duration_seconds: 0,
      level_db: -60,
    });
    apiMock.dictate.mockRejectedValue(new TimedOutError(600_000, "/pipeline/dictate"));

    const widget = document.getElementById("widget")!;
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.getElementById("widget-text")!.textContent).toBe("Recording");
    });
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => {
      expect(document.getElementById("widget-text")!.textContent).toBe("No answer");
    });

    await vi.advanceTimersByTimeAsync(60_000);

    expect(document.getElementById("widget-text")!.textContent).toBe("No answer");
  });
});
