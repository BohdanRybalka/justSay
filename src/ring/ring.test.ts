// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  EVENT_DICTATION_TOGGLE,
  EVENT_MEETING_TOGGLE,
  EVENT_NAVIGATE_PANEL,
  EVENT_RING_CLOSED,
  EVENT_RING_OPENED,
  EVENT_RING_POINTER,
  EVENT_SETTINGS_CHANGED,
} from "../contracts";
import { BUTTON_HOVER_CLASS, RING_OPEN_CLASS } from "./ring-view";

const listeners = new Map<string, (event: unknown) => unknown>();

const SETTINGS = {
  theme: "dark",
  language: "uk",
  previous_language: "en",
  meeting_consent_acknowledged: true,
  shortcut: "Ctrl+Alt+KeyV",
  meetings_language: "uk",
  files_language: "auto",
};

const { invokeMock, emitMock, getSettingsMock, getMeetingStatusMock, updateSettingsMock, notifyErrorMock } = vi.hoisted(() => ({
  notifyErrorMock: vi.fn(async (_body: string) => {}),
  invokeMock: vi.fn(async (_command: string) => {}),
  emitMock: vi.fn(async (_event: string, _payload?: unknown) => {}),
  getSettingsMock: vi.fn(async (): Promise<Record<string, unknown>> => ({})),
  getMeetingStatusMock: vi.fn(async () => ({ is_recording: false })),
  updateSettingsMock: vi.fn(async (_updates: Record<string, unknown>) => ({})),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("../notify", () => ({ notifyError: notifyErrorMock }));
vi.mock("@tauri-apps/api/event", () => ({
  emit: emitMock,
  listen: vi.fn(async (event: string, handler: (payload: unknown) => unknown) => {
    listeners.set(event, handler);
    return () => {};
  }),
}));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: {
    getSettings: getSettingsMock,
    getMeetingStatus: getMeetingStatusMock,
    updateSettings: updateSettingsMock,
  },
}));

async function loadRing(): Promise<HTMLElement> {
  const html = readFileSync(resolve(__dirname, "../../ring.html"), "utf-8");
  document.body.innerHTML = html
    .slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"))
    .replace(/<script[\s\S]*?<\/script>/g, "");
  await import("./ring");
  await vi.waitFor(() => expect(listeners.get(EVENT_SETTINGS_CHANGED)).toBeTypeOf("function"));
  return document.getElementById("ring")!;
}

function shell(event: string, payload?: unknown) {
  listeners.get(event)!({ payload });
}

const BUTTON = { meeting: 0, file: 1, dictate: 2, language: 3, settings: 4 } as const;

function button(ring: HTMLElement, action: keyof typeof BUTTON): HTMLElement {
  return ring.querySelectorAll<HTMLElement>(".ring-button")[BUTTON[action]];
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getSettingsMock.mockResolvedValue(SETTINGS);
  getMeetingStatusMock.mockResolvedValue({ is_recording: false });
  listeners.clear();
  delete document.documentElement.dataset.theme;
  vi.stubGlobal("matchMedia", () => ({
    matches: false,
    addEventListener: () => {},
    removeEventListener: () => {},
  }));
});

describe("the ring window", () => {
  it("slides in when the shell opens it and rests when the shell closes it", async () => {
    const ring = await loadRing();

    shell(EVENT_RING_OPENED);
    expect(ring.classList.contains(RING_OPEN_CLASS)).toBe(true);

    shell(EVENT_RING_CLOSED);
    expect(ring.classList.contains(RING_OPEN_CLASS)).toBe(false);
  });

  it("hovers the button the shell reports the pointer over", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);

    shell(EVENT_RING_POINTER, { x: -19, y: -60 });

    expect(button(ring, "meeting").classList.contains(BUTTON_HOVER_CLASS)).toBe(true);
  });

  it("asks the shell to close it when a button is picked", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);

    button(ring, "settings").click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("close_ring"));
  });

  it("asks the shell to close it on a right click on the mic, doing nothing else", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);

    button(ring, "dictate").dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2 }));

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("close_ring"));
    expect(emitMock).not.toHaveBeenCalled();
  });

  it("never shows the webview's own menu on a right-click inside the ring", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);
    const rightClick = new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2 });

    button(ring, "meeting").dispatchEvent(rightClick);

    expect(rightClick.defaultPrevented).toBe(true);
  });

  it("reads the theme again on opening when the backend was not up at launch", async () => {
    getSettingsMock.mockRejectedValueOnce(new Error("connection refused"));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await loadRing();
    await vi.waitFor(() => expect(getSettingsMock).toHaveBeenCalledTimes(1));
    expect(document.documentElement.dataset.theme).toBe("light");

    shell(EVENT_RING_OPENED);

    await vi.waitFor(() => expect(document.documentElement.dataset.theme).toBe("dark"));
  });

  it("takes the theme the settings name, and the next one chosen in Settings", async () => {
    await loadRing();
    await vi.waitFor(() => expect(document.documentElement.dataset.theme).toBe("dark"));

    getSettingsMock.mockResolvedValueOnce({ ...SETTINGS, theme: "light" });
    shell(EVENT_SETTINGS_CHANGED);

    await vi.waitFor(() => expect(document.documentElement.dataset.theme).toBe("light"));
  });

  it("offers the mic with the shortcut that dictates without the ring", async () => {
    const ring = await loadRing();

    await vi.waitFor(() =>
      expect(button(ring, "dictate").getAttribute("aria-label")).toBe("Dictate · Ctrl + Alt + V"),
    );
  });

  it("hands a dictation to the widget, which owns dictation", async () => {
    const ring = await loadRing();

    button(ring, "dictate").click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_DICTATION_TOGGLE));
    expect(invokeMock).toHaveBeenCalledWith("close_ring");
  });

  it("names the language in use and offers to stop a meeting being recorded", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(button(ring, "language").getAttribute("aria-label")).toBe("Language · Ukrainian"));

    getMeetingStatusMock.mockResolvedValueOnce({ is_recording: true });
    shell(EVENT_RING_OPENED);

    await vi.waitFor(() => expect(button(ring, "meeting").getAttribute("aria-label")).toBe("Stop the meeting"));
  });

  it("hands a meeting to the widget, which owns the meeting flow", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(getMeetingStatusMock).toHaveBeenCalled());

    button(ring, "meeting").click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_MEETING_TOGGLE));
  });

  it("opens the Meetings page in the main window until the disclosure is acknowledged", async () => {
    getSettingsMock.mockResolvedValue({ ...SETTINGS, meeting_consent_acknowledged: false });
    const ring = await loadRing();
    await vi.waitFor(() => expect(button(ring, "language").getAttribute("aria-label")).toBe("Language · Ukrainian"));

    button(ring, "meeting").click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("show_settings_window"));
    expect(emitMock).toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, { panel: "meetings" });
    expect(emitMock).not.toHaveBeenCalledWith(EVENT_MEETING_TOGGLE);
  });

  it("switches to the language used before and tells the other windows", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(button(ring, "language").getAttribute("aria-label")).toBe("Language · Ukrainian"));

    button(ring, "language").click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_SETTINGS_CHANGED));
    expect(updateSettingsMock).toHaveBeenCalledWith({ language: "en" });
  });

  it("asks the shell to open the file dialog for a file", async () => {
    const ring = await loadRing();

    button(ring, "file").click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("pick_audio_file"));
    expect(emitMock).not.toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, expect.anything());
  });

  it("opens the main window on Settings for Settings", async () => {
    const ring = await loadRing();

    button(ring, "settings").click();
    await vi.waitFor(() =>
      expect(emitMock).toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, { panel: "settings" }),
    );
    expect(invokeMock).toHaveBeenCalledWith("show_settings_window");
  });

  it("says in a notification that the language did not switch, the ring being closed by then", async () => {
    updateSettingsMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const ring = await loadRing();
    await vi.waitFor(() => expect(button(ring, "language").getAttribute("aria-label")).toBe("Language · Ukrainian"));

    button(ring, "language").click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("Couldn't switch to English — try again."));
    expect(invokeMock).toHaveBeenCalledWith("close_ring");
    expect(emitMock).not.toHaveBeenCalledWith(EVENT_SETTINGS_CHANGED);
  });

  it("raises no notification for a button that worked", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(button(ring, "language").getAttribute("aria-label")).toBe("Language · Ukrainian"));

    button(ring, "language").click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_SETTINGS_CHANGED));
    expect(notifyErrorMock).not.toHaveBeenCalled();
  });
});
