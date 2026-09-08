// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EVENT_MEETING_TOGGLE } from "../contracts";
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
  audioDiscard: vi.fn(),
  audioStatus: vi.fn(),
  dictate: vi.fn(),
};

const SESSION_PATTERN = /^[0-9a-f]{32}$/;

/** The id the widget minted for the start it has just issued. Read back off
 *  the mock rather than injected, because the whole point of the design is
 *  that the client chooses the name and nothing else gets to. */
function mintedSession(): string {
  return apiMock.audioStart.mock.calls[0][0] as string;
}

/** `vi.resetModules()` runs before every test, so the widget under test holds a
 *  *different* copy of `../timeout` than a static import here would. An
 *  `instanceof` against the wrong copy is always false, and the branch being
 *  tested is chosen by exactly that check — so the error is built from the same
 *  module instance the widget loaded. */
async function timedOut(subject: string) {
  const { TimedOutError } = await import("../timeout");
  return new TimedOutError(15_000, subject);
}

/** How long the widget may sit in `processing` against a backend that accepts
 *  a dictation and never answers: the probe's 15 s eligibility wait, one 5 s
 *  poll period, and the probe's own 15 s budget. */
const PROCESSING_CEILING_MS = 35_000;

async function advanceUntil(predicate: () => boolean, budgetMs: number): Promise<number> {
  for (let elapsed = 0; elapsed <= budgetMs; elapsed += 1_000) {
    if (predicate()) return elapsed;
    await vi.advanceTimersByTimeAsync(1_000);
  }
  return Number.POSITIVE_INFINITY;
}

function recordingStatus(overrides: Record<string, unknown> = {}) {
  return {
    is_recording: true,
    duration_seconds: 0,
    level_db: -60,
    session_id: null,
    ...overrides,
  };
}

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

const { invokeMock } = vi.hoisted(() => ({ invokeMock: vi.fn(async () => {}) }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
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
  invokeMock.mockResolvedValue(undefined);
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

  it("keeps probing while one is unanswered, and lets only the newest answer speak", async () => {
    await loadWidget();
    const pending: Array<(ok: boolean) => void> = [];
    apiMock.health.mockImplementation(
      () =>
        new Promise((resolve, reject) => {
          pending.push((ok) =>
            ok
              ? resolve({ status: "ok", version: "0", stt_mode: "cloud", llm_mode: "cloud" })
              : reject(new TypeError("Failed to fetch")),
          );
        }),
    );
    const before = apiMock.health.mock.calls.length;

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 4);

    expect(apiMock.health.mock.calls.length - before).toBe(4);

    pending[pending.length - 1](true);
    await vi.advanceTimersByTimeAsync(0);
    for (const settle of pending.slice(0, -1)) settle(false);
    await vi.advanceTimersByTimeAsync(0);

    expect(document.getElementById("widget-text")!.textContent).toBe("JustSay");
    expect(notifyErrorMock).not.toHaveBeenCalledWith("JustSay backend is unreachable.");
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

describe("two dictations finishing within three seconds of each other", () => {
  it("does not let the first one's auto-revert cut the second one's result short", async () => {
    await loadWidget();
    apiMock.audioStart.mockResolvedValue(recordingStatus());
    apiMock.dictate.mockResolvedValue({
      text: "one",
      duration_ms: 100,
      copied_to_clipboard: true,
    });
    const widget = document.getElementById("widget")!;
    const text = document.getElementById("widget-text")!;

    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Recording"));
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Copied"));

    await vi.advanceTimersByTimeAsync(1500);
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Recording"));
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Copied"));

    await vi.advanceTimersByTimeAsync(1600);

    expect(text.textContent).toBe("Copied");

    await vi.advanceTimersByTimeAsync(1500);

    expect(text.textContent).toBe("JustSay");
  });
});

describe("a Tauri bridge that stops answering", () => {
  it("does not leave the meeting toggle swallowing every later press", async () => {
    await loadWidget();
    await vi.waitFor(() => expect(listeners.get(EVENT_MEETING_TOGGLE)).toBeTypeOf("function"));
    const pressTray = listeners.get(EVENT_MEETING_TOGGLE)!;
    apiMock.startMeetingRecording.mockResolvedValue({
      is_recording: true,
      duration_seconds: 0,
      level_db: -60,
      system_endpoint: null,
      system_level_db: -60,
    });
    apiMock.stopMeetingRecording.mockResolvedValue({ path: "meeting.wav" });
    invokeMock.mockImplementation(() => new Promise(() => {}));

    void pressTray({});
    await vi.advanceTimersByTimeAsync(0);
    expect(apiMock.startMeetingRecording).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(3000);

    void pressTray({});
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.stopMeetingRecording).toHaveBeenCalledOnce();
  });
});

describe("a start that runs out of its budget", () => {
  it("adopts the recording when the backend names this window's own session", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockImplementation(async () =>
      recordingStatus({ duration_seconds: 12, session_id: mintedSession() }),
    );

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(apiMock.audioStatus).toHaveBeenCalled());
    await vi.advanceTimersByTimeAsync(0);

    expect(document.getElementById("widget-text")!.textContent).toBe("Recording");
    expect(document.getElementById("widget-duration")!.textContent).toBe("12.0s");
    expect(apiMock.audioDiscard).not.toHaveBeenCalled();
  });

  it("mints a session id of the shape the backend validates against", async () => {
    await loadWidget();
    apiMock.audioStart.mockResolvedValue(recordingStatus());

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(apiMock.audioStart).toHaveBeenCalledOnce());

    expect(mintedSession()).toMatch(SESSION_PATTERN);
  });

  it("still owes the session when the status names somebody else", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ session_id: "ffffffffffffffffffffffffffffffff" }),
    );
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 0 });
    const { REQUEST_TIMEOUT_MS } = await import("../api");

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() =>
      expect(document.getElementById("widget-text")!.textContent).toBe("Start failed"),
    );

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + CONNECTION_POLL_MS * 2);

    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());
  });

  it("discards the capture when the named owner stops and the queued start lands", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ session_id: "ffffffffffffffffffffffffffffffff" }),
    );
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 7 });
    const { REQUEST_TIMEOUT_MS } = await import("../api");

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() =>
      expect(document.getElementById("widget-text")!.textContent).toBe("Start failed"),
    );

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + CONNECTION_POLL_MS * 2);

    expect(apiMock.audioDiscard).toHaveBeenCalledTimes(1);
    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());

    await vi.advanceTimersByTimeAsync(CONNECTION_POLL_MS * 3);

    expect(apiMock.audioDiscard).toHaveBeenCalledTimes(1);
  });

  it("owes the session when the status is idle, because the start may still be queued", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockResolvedValue(
      recordingStatus({ is_recording: false, session_id: null }),
    );
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 3 });
    const { REQUEST_TIMEOUT_MS } = await import("../api");

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() =>
      expect(document.getElementById("widget-text")!.textContent).toBe("Start failed"),
    );

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + CONNECTION_POLL_MS * 2);

    expect(apiMock.audioDiscard).toHaveBeenCalledTimes(1);
    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());
  });

  it("waits from the moment the start was issued, so the status read's budget does not stack", async () => {
    await loadWidget();
    const { REQUEST_TIMEOUT_MS } = await import("../api");
    const started = await timedOut("/audio/start");
    const read = await timedOut("/audio/status");
    apiMock.audioStart.mockImplementation(
      () =>
        new Promise((_resolve, reject) => setTimeout(() => reject(started), REQUEST_TIMEOUT_MS)),
    );
    apiMock.audioStatus.mockImplementation(
      () => new Promise((_resolve, reject) => setTimeout(() => reject(read), REQUEST_TIMEOUT_MS)),
    );
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 0 });

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS * 2 + CONNECTION_POLL_MS * 2);

    expect(apiMock.audioDiscard).toHaveBeenCalledTimes(1);
    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());
  });

  it("owes the session when the status read fails, and probes it against a dead /health", async () => {
    await loadWidget();
    apiMock.audioStart.mockRejectedValue(await timedOut("/audio/start"));
    apiMock.audioStatus.mockRejectedValue(await timedOut("/audio/status"));
    apiMock.audioDiscard.mockImplementation(() => new Promise(() => {}));
    apiMock.health.mockImplementation(() => new Promise(() => {}));
    const { REQUEST_TIMEOUT_MS } = await import("../api");

    document.getElementById("widget")!.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() =>
      expect(document.getElementById("widget-text")!.textContent).toBe("Start failed"),
    );

    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + CONNECTION_POLL_MS * 2);

    expect(apiMock.audioDiscard).toHaveBeenCalledTimes(1);
    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());
  });
});

describe("a dictation the backend accepts and never answers", () => {
  it("leaves processing within 35 seconds and discards the capture", async () => {
    await loadWidget();
    apiMock.audioStart.mockResolvedValue(recordingStatus());
    apiMock.dictate.mockImplementation(() => new Promise(() => {}));
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 4 });
    const widget = document.getElementById("widget")!;
    const text = document.getElementById("widget-text")!;

    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Recording"));
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Processing"));

    const elapsed = await advanceUntil(
      () => text.textContent === "No answer",
      PROCESSING_CEILING_MS,
    );

    expect(elapsed).toBeLessThanOrEqual(PROCESSING_CEILING_MS);
    expect(apiMock.audioDiscard).toHaveBeenCalledWith(mintedSession());
  });

  it("keeps waiting while the probe has not yet come due", async () => {
    await loadWidget();
    apiMock.audioStart.mockResolvedValue(recordingStatus());
    apiMock.dictate.mockImplementation(() => new Promise(() => {}));
    apiMock.audioDiscard.mockResolvedValue({ duration_seconds: 4 });
    const widget = document.getElementById("widget")!;
    const text = document.getElementById("widget-text")!;

    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Recording"));
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Processing"));

    await vi.advanceTimersByTimeAsync(14_000);

    expect(apiMock.audioDiscard).not.toHaveBeenCalled();
    expect(text.textContent).toBe("Processing");
  });

  it("owes nothing once the dictation answers", async () => {
    await loadWidget();
    apiMock.audioStart.mockResolvedValue(recordingStatus());
    apiMock.dictate.mockResolvedValue({
      text: "one",
      duration_ms: 100,
      copied_to_clipboard: true,
    });
    const widget = document.getElementById("widget")!;
    const text = document.getElementById("widget-text")!;

    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Recording"));
    widget.dispatchEvent(new MouseEvent("click"));
    await vi.waitFor(() => expect(text.textContent).toBe("Copied"));

    await vi.advanceTimersByTimeAsync(40_000);

    expect(apiMock.audioDiscard).not.toHaveBeenCalled();
  });
});
