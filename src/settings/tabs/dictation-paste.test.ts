// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const { invokeMock, saveSettingsMock, notifyErrorMock } = vi.hoisted(() => ({
  invokeMock: vi.fn<(command: string) => Promise<unknown>>(),
  saveSettingsMock: vi.fn(),
  notifyErrorMock: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("../settings", () => ({ saveSettings: saveSettingsMock }));
vi.mock("../../notify", () => ({ notifyError: notifyErrorMock }));

const { PASTE_ROW, PERMISSION_HINT, wirePasteRow } = await import("./dictation-paste");

const PASTE_HINT = "Text lands at the cursor as soon as it's ready";

let permitted = true;
let teardown: (() => void) | null = null;

beforeEach(() => {
  vi.resetAllMocks();
  permitted = true;
  invokeMock.mockImplementation(async (command) =>
    command === "paste_permission_granted" ? permitted : undefined,
  );
  saveSettingsMock.mockResolvedValue({});
});

afterEach(() => {
  teardown?.();
  teardown = null;
});

function mount(pasteAtCursor: boolean, onSaved = vi.fn()) {
  const container = document.createElement("div");
  container.innerHTML = PASTE_ROW;
  teardown = wirePasteRow(container, { paste_at_cursor: pasteAtCursor } as UserSettings, onSaved);
  return {
    toggle: container.querySelector<HTMLButtonElement>("#paste-toggle")!,
    hint: container.querySelector<HTMLElement>("#paste-hint")!,
    allow: container.querySelector<HTMLButtonElement>("#btn-allow-paste")!,
  };
}

describe("Paste where I'm typing", () => {
  it("shows the saved choice on the switch", () => {
    expect(mount(true).toggle.getAttribute("aria-checked")).toBe("true");
    teardown?.();
    expect(mount(false).toggle.getAttribute("aria-checked")).toBe("false");
  });

  it("saves the switch and tells the widget", async () => {
    const onSaved = vi.fn();
    const { toggle } = mount(true, onSaved);

    toggle.click();

    await vi.waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(saveSettingsMock).toHaveBeenCalledExactlyOnceWith({ paste_at_cursor: false });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });

  it("puts the switch back when the save fails", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const onSaved = vi.fn();
    const { toggle } = mount(true, onSaved);

    toggle.click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    await vi.waitFor(() => expect(toggle.disabled).toBe(false));
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("says where to allow key events while the system has not, and opens that pane", async () => {
    permitted = false;
    const { hint, allow } = mount(true);

    await vi.waitFor(() => expect(hint.textContent).toBe(PERMISSION_HINT));
    expect(hint.classList.contains("setting-row-hint--result")).toBe(true);
    expect(allow.hidden).toBe(false);

    allow.click();
    await vi.waitFor(() =>
      expect(invokeMock).toHaveBeenCalledWith("open_accessibility_settings"),
    );
  });

  it("asks for nothing while pasting is off", async () => {
    permitted = false;
    const { hint, allow } = mount(false);

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("paste_permission_granted"));
    await Promise.resolve();
    expect(hint.textContent).toBe(PASTE_HINT);
    expect(allow.hidden).toBe(true);
  });

  it("drops the hint once the permission is given and the window is back in focus", async () => {
    permitted = false;
    const { hint, allow } = mount(true);
    await vi.waitFor(() => expect(hint.textContent).toBe(PERMISSION_HINT));

    permitted = true;
    window.dispatchEvent(new Event("focus"));

    await vi.waitFor(() => expect(hint.textContent).toBe(PASTE_HINT));
    expect(hint.classList.contains("setting-row-hint--result")).toBe(false);
    expect(allow.hidden).toBe(true);
  });

  it("stops checking once the row is gone", async () => {
    mount(true);
    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledOnce());

    teardown?.();
    teardown = null;
    window.dispatchEvent(new Event("focus"));
    await Promise.resolve();

    expect(invokeMock).toHaveBeenCalledOnce();
  });
});
