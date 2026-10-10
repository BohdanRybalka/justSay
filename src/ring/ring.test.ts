// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  EVENT_MEETING_TOGGLE,
  EVENT_NAVIGATE_PANEL,
  EVENT_RING_CLOSED,
  EVENT_RING_OPENED,
  EVENT_RING_POINTER,
  EVENT_SETTINGS_CHANGED,
} from "../contracts";
import { PETAL_HOVER_CLASS, RING_OPEN_CLASS } from "./ring-view";

const listeners = new Map<string, (event: unknown) => unknown>();

const SETTINGS = {
  theme: "dark",
  language: "uk",
  previous_language: "en",
  meeting_consent_acknowledged: true,
  meetings_engine: "local",
  meetings_language: "uk",
};

const { invokeMock, emitMock, getSettingsMock, getMeetingStatusMock, updateSettingsMock } = vi.hoisted(() => ({
  invokeMock: vi.fn(async (_command: string) => {}),
  emitMock: vi.fn(async (_event: string, _payload?: unknown) => {}),
  getSettingsMock: vi.fn(async (): Promise<Record<string, unknown>> => ({})),
  getMeetingStatusMock: vi.fn(async () => ({ is_recording: false })),
  updateSettingsMock: vi.fn(async (_updates: Record<string, unknown>) => ({})),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
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

function petal(ring: HTMLElement, index: number): HTMLElement {
  return ring.querySelectorAll<HTMLElement>(".ring-petal")[index];
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
  it("fans out when the shell opens it and rests when the shell closes it", async () => {
    const ring = await loadRing();

    shell(EVENT_RING_OPENED);
    expect(ring.classList.contains(RING_OPEN_CLASS)).toBe(true);

    shell(EVENT_RING_CLOSED);
    expect(ring.classList.contains(RING_OPEN_CLASS)).toBe(false);
  });

  it("hovers the petal the shell reports the pointer over", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);

    shell(EVENT_RING_POINTER, { x: 0, y: -38 });

    expect(ring.querySelectorAll(".ring-petal")[0].classList.contains(PETAL_HOVER_CLASS)).toBe(true);
  });

  it("asks the shell to close it when a petal is picked", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);

    ring.querySelectorAll<HTMLElement>(".ring-petal")[3].click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("close_ring"));
  });

  it("never shows the webview's own menu on a right-click inside the ring", async () => {
    const ring = await loadRing();
    shell(EVENT_RING_OPENED);
    const rightClick = new MouseEvent("contextmenu", { bubbles: true, cancelable: true, button: 2 });

    ring.querySelectorAll<HTMLElement>(".ring-petal")[0].dispatchEvent(rightClick);

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

  it("names the language in use and offers to stop a meeting being recorded", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(petal(ring, 2).getAttribute("aria-label")).toBe("Language · Ukrainian"));

    getMeetingStatusMock.mockResolvedValueOnce({ is_recording: true });
    shell(EVENT_RING_OPENED);

    await vi.waitFor(() => expect(petal(ring, 0).getAttribute("aria-label")).toBe("Stop the meeting"));
  });

  it("hands a meeting to the widget, which owns the meeting flow", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(getMeetingStatusMock).toHaveBeenCalled());

    petal(ring, 0).click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_MEETING_TOGGLE));
  });

  it("opens the Meetings page in the main window until the disclosure is acknowledged", async () => {
    getSettingsMock.mockResolvedValue({ ...SETTINGS, meeting_consent_acknowledged: false });
    const ring = await loadRing();
    await vi.waitFor(() => expect(petal(ring, 2).getAttribute("aria-label")).toBe("Language · Ukrainian"));

    petal(ring, 0).click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("show_settings_window"));
    expect(emitMock).toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, { panel: "meetings" });
    expect(emitMock).not.toHaveBeenCalledWith(EVENT_MEETING_TOGGLE);
  });

  it("switches to the language used before and tells the other windows", async () => {
    const ring = await loadRing();
    await vi.waitFor(() => expect(petal(ring, 2).getAttribute("aria-label")).toBe("Language · Ukrainian"));

    petal(ring, 2).click();

    await vi.waitFor(() => expect(emitMock).toHaveBeenCalledWith(EVENT_SETTINGS_CHANGED));
    expect(updateSettingsMock).toHaveBeenCalledWith({ language: "en" });
  });

  it("asks the shell to open the file dialog for a file", async () => {
    const ring = await loadRing();

    petal(ring, 1).click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("pick_audio_file"));
    expect(emitMock).not.toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, expect.anything());
  });

  it("opens the main window on Settings for Settings", async () => {
    const ring = await loadRing();

    petal(ring, 3).click();
    await vi.waitFor(() =>
      expect(emitMock).toHaveBeenCalledWith(EVENT_NAVIGATE_PANEL, { panel: "settings" }),
    );
    expect(invokeMock).toHaveBeenCalledWith("show_settings_window");
  });
});
