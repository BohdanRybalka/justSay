// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  EVENT_RING_CLOSED,
  EVENT_RING_OPENED,
  EVENT_RING_POINTER,
  EVENT_SETTINGS_CHANGED,
} from "../contracts";
import { PETAL_HOVER_CLASS, RING_OPEN_CLASS } from "./ring-view";

const listeners = new Map<string, (event: unknown) => unknown>();

const { invokeMock, getSettingsMock } = vi.hoisted(() => ({
  invokeMock: vi.fn(async (_command: string) => {}),
  getSettingsMock: vi.fn(async () => ({ theme: "dark" })),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (payload: unknown) => unknown) => {
    listeners.set(event, handler);
    return () => {};
  }),
}));
vi.mock("../api", () => ({ api: { getSettings: getSettingsMock } }));

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

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
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

    getSettingsMock.mockResolvedValueOnce({ theme: "light" });
    shell(EVENT_SETTINGS_CHANGED);

    await vi.waitFor(() => expect(document.documentElement.dataset.theme).toBe("light"));
  });
});
