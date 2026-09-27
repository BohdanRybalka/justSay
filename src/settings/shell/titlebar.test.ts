// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderTitlebar, wireTitlebar } from "./titlebar";

let maximised = false;
let resizeHandler: (() => void) | null = null;

const fakeWindow = {
  minimize: vi.fn(async () => {}),
  toggleMaximize: vi.fn(async () => {}),
  close: vi.fn(async () => {}),
  isMaximized: vi.fn(async () => maximised),
  onResized: vi.fn(async (handler: () => void) => {
    resizeHandler = handler;
    return () => {};
  }),
};

function newBar(): HTMLElement {
  document.body.innerHTML = '<header id="titlebar" class="titlebar"></header>';
  return document.getElementById("titlebar")!;
}

function click(bar: HTMLElement, selector: string): void {
  bar.querySelector<HTMLButtonElement>(selector)!.click();
}

beforeEach(() => {
  maximised = false;
  resizeHandler = null;
  vi.clearAllMocks();
});

describe("renderTitlebar", () => {
  it("makes the bar, the mark and the name the drag region", () => {
    const bar = newBar();
    renderTitlebar(bar, "windows");

    expect(bar.hasAttribute("data-tauri-drag-region")).toBe(true);
    expect(bar.querySelector(".brand-mark")!.hasAttribute("data-tauri-drag-region")).toBe(true);
    const name = bar.querySelector(".titlebar-name")!;
    expect(name.textContent).toBe("JustSay");
    expect(name.hasAttribute("data-tauri-drag-region")).toBe(true);
  });

  it("draws minimise, maximise and close on Windows, outside the drag region", () => {
    const bar = newBar();
    renderTitlebar(bar, "windows");

    const buttons = [...bar.querySelectorAll("button")];
    expect(buttons.map((button) => button.getAttribute("aria-label"))).toEqual([
      "Minimise",
      "Maximise",
      "Close",
    ]);
    expect(buttons.some((button) => button.hasAttribute("data-tauri-drag-region"))).toBe(false);
  });

  it("leaves the controls to the native traffic lights on macOS", () => {
    const bar = newBar();
    renderTitlebar(bar, "mac");

    expect(bar.querySelectorAll("button")).toHaveLength(0);
    expect(bar.classList.contains("titlebar--mac")).toBe(true);
  });
});

describe("wireTitlebar", () => {
  it("minimises, toggles maximise and closes the window from its buttons", async () => {
    const bar = newBar();
    renderTitlebar(bar, "windows");
    await wireTitlebar(bar, fakeWindow);

    click(bar, "#btn-minimise");
    click(bar, "#btn-maximise");
    click(bar, "#btn-close");

    expect(fakeWindow.minimize).toHaveBeenCalledTimes(1);
    expect(fakeWindow.toggleMaximize).toHaveBeenCalledTimes(1);
    expect(fakeWindow.close).toHaveBeenCalledTimes(1);
  });

  it("turns maximise into restore while the window is maximised", async () => {
    const bar = newBar();
    renderTitlebar(bar, "windows");
    maximised = true;
    await wireTitlebar(bar, fakeWindow);

    const button = bar.querySelector("#btn-maximise")!;
    expect(button.getAttribute("aria-label")).toBe("Restore");
    expect(button.querySelector("use")!.getAttribute("href")).toBe("#restore");

    maximised = false;
    resizeHandler!();
    await vi.waitFor(() => expect(button.getAttribute("aria-label")).toBe("Maximise"));
    expect(button.querySelector("use")!.getAttribute("href")).toBe("#sq");
  });

  it("asks nothing of the window on macOS", async () => {
    const bar = newBar();
    renderTitlebar(bar, "mac");
    await wireTitlebar(bar, fakeWindow);

    expect(fakeWindow.onResized).not.toHaveBeenCalled();
    expect(fakeWindow.isMaximized).not.toHaveBeenCalled();
  });
});
