// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RecordingStatus } from "../api";
import { EVENT_MEETING_TOGGLE } from "../contracts";
import { MEETING_STATE_CLASS } from "./meeting-indicator";
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

function meetingStatus(isRecording: boolean, durationSeconds = 0) {
  return {
    is_recording: isRecording,
    duration_seconds: durationSeconds,
    level_db: -60,
    system_endpoint: null,
    system_level_db: -60,
  };
}

async function triggerMeetingToggle() {
  await vi.waitFor(() => expect(listeners.has(EVENT_MEETING_TOGGLE)).toBe(true));
  await listeners.get(EVENT_MEETING_TOGGLE)!({});
}

function meetingIndicatorShown(): boolean {
  return root().classList.contains(MEETING_STATE_CLASS);
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

describe("the stop a widget owes after a start it could not adopt", () => {
  async function abandonStart() {
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockRejectedValue(await timedOut("/audio/status"));
    apiMock.audioStop.mockResolvedValue({ filename: "rec.wav", duration_seconds: 1 });
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));
  }

  it("issues exactly one stop on the first poll that reaches a live backend", async () => {
    await abandonStart();
    expect(apiMock.audioStop).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS);
    await vi.waitFor(() => expect(apiMock.audioStop).toHaveBeenCalledOnce());

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(apiMock.audioStop).toHaveBeenCalledOnce();
  });

  it("holds the stop while the backend is unreachable and pays it when it answers", async () => {
    await abandonStart();
    apiMock.health.mockRejectedValue(new Error("backend unreachable"));

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 6);
    expect(apiMock.audioStop).not.toHaveBeenCalled();

    apiMock.health.mockResolvedValue({
      status: "ok",
      version: "0",
      stt_mode: "cloud",
      llm_mode: "cloud",
    });
    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS);
    await vi.waitFor(() => expect(apiMock.audioStop).toHaveBeenCalledOnce());
  });

  it("owes nothing when the recording was adopted", async () => {
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ is_recording: true, duration_seconds: 7.5 }),
    );
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(apiMock.audioStatus).toHaveBeenCalled());

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(apiMock.audioStop).not.toHaveBeenCalled();
  });
});

describe("the stop a widget owes after a dictation the backend never answered", () => {
  async function abandonDictate() {
    apiMock.audioStart.mockResolvedValue(recordingStatus({ is_recording: true }));
    apiMock.dictate.mockRejectedValue(await timedOut("/pipeline/dictate"));
    apiMock.audioStop.mockResolvedValue({ filename: "rec.wav", duration_seconds: 1 });
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget recording"));
    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));
  }

  it("issues one stop on the next healthy poll when the status read cannot answer either", async () => {
    apiMock.audioStatus.mockRejectedValue(await timedOut("/audio/status"));
    await abandonDictate();

    expect(apiMock.audioStatus).toHaveBeenCalledOnce();
    expect(apiMock.audioStop).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS);
    await vi.waitFor(() => expect(apiMock.audioStop).toHaveBeenCalledOnce());
  });

  it("owes nothing when the backend positively reports the microphone closed", async () => {
    apiMock.audioStatus.mockResolvedValue(recordingStatus({ is_recording: false }));
    await abandonDictate();

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(apiMock.audioStop).not.toHaveBeenCalled();
  });

  it("reads no status at all when the dictation fails for an ordinary reason", async () => {
    apiMock.audioStart.mockResolvedValue(recordingStatus({ is_recording: true }));
    apiMock.dictate.mockRejectedValue(new Error("connection refused"));
    await loadWidget();

    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget recording"));
    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(root().className).toBe("widget error"));

    expect(apiMock.audioStatus).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(apiMock.audioStop).not.toHaveBeenCalled();
  });
});

describe("a meeting indicator raised on a status the widget could not read", () => {
  async function abandonMeetingStart() {
    apiMock.startMeetingRecording.mockRejectedValue(await timedOut("/audio/meeting/start"));
    apiMock.getMeetingStatus.mockRejectedValue(new Error("no answer"));
    await loadWidget();
    await triggerMeetingToggle();
    expect(meetingIndicatorShown()).toBe(true);
  }

  it("comes back down on the first poll that reads no recording, with no user action", async () => {
    await abandonMeetingStart();

    apiMock.getMeetingStatus.mockResolvedValue(meetingStatus(false));
    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS);
    await vi.waitFor(() => expect(meetingIndicatorShown()).toBe(false));
  });

  it("stays up when the poll finds the call really is being recorded", async () => {
    await abandonMeetingStart();

    apiMock.getMeetingStatus.mockResolvedValue(meetingStatus(true, 12));
    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(meetingIndicatorShown()).toBe(true);
  });

  it("keeps one confirmation read in flight while the backend stays silent", async () => {
    await abandonMeetingStart();
    apiMock.getMeetingStatus.mockImplementation(() => new Promise(() => {}));
    const before = apiMock.getMeetingStatus.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);

    expect(apiMock.getMeetingStatus.mock.calls.length - before).toBe(1);
  });

  it("is not re-read once the status has confirmed it", async () => {
    await abandonMeetingStart();

    apiMock.getMeetingStatus.mockResolvedValue(meetingStatus(true, 12));
    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS);
    await vi.waitFor(() => expect(apiMock.getMeetingStatus).toHaveBeenCalled());
    const reads = apiMock.getMeetingStatus.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);
    expect(apiMock.getMeetingStatus).toHaveBeenCalledTimes(reads);
  });
});

describe("the widget's own timers", () => {
  it("leaves exactly one stopwatch running after an adopted recording", async () => {
    apiMock.audioStart.mockResolvedValue(recordingStatus({ is_recording: true }));
    await loadWidget();
    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(apiMock.audioStart).toHaveBeenCalled());
    const afterAPlainStart = vi.getTimerCount();

    vi.useRealTimers();
    vi.useFakeTimers();
    vi.resetModules();
    listeners.clear();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ is_recording: true, duration_seconds: 7.5 }),
    );
    await loadWidget();
    root().dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() =>
      expect(document.getElementById("widget-duration")!.textContent).toMatch(/^7\.\ds$/),
    );

    expect(vi.getTimerCount()).toBe(afterAPlainStart);
  });

  it("advances only its own connection poll, whatever ran before it", async () => {
    await loadWidget();
    const before = apiMock.health.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);

    expect(apiMock.health.mock.calls.length - before).toBe(3);
  });
});
