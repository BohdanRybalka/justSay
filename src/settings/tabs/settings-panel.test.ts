// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const { saveSettingsMock, emitSettingsChangedMock, applyAppThemeMock, notifyErrorMock } = vi.hoisted(
  () => ({
    saveSettingsMock: vi.fn(),
    emitSettingsChangedMock: vi.fn(async () => {}),
    applyAppThemeMock: vi.fn(async () => {}),
    notifyErrorMock: vi.fn(async () => {}),
  }),
);

vi.mock("../settings", () => ({ saveSettings: saveSettingsMock }));
vi.mock("./general", () => ({ emitSettingsChanged: emitSettingsChangedMock }));
vi.mock("../../ui/theme", () => ({ applyAppTheme: applyAppThemeMock }));
vi.mock("../../notify", () => ({ notifyError: notifyErrorMock }));

import { renderSettingsPanel } from "./settings-panel";

function render(theme: UserSettings["theme"]): HTMLElement {
  const container = document.createElement("div");
  document.body.replaceChildren(container);
  renderSettingsPanel(container, { theme } as UserSettings);
  return container;
}

function themeButton(container: HTMLElement, label: string): HTMLButtonElement {
  return [...container.querySelectorAll<HTMLButtonElement>(".theme-choice button")].find(
    (button) => button.textContent === label,
  )!;
}

function pressedTheme(container: HTMLElement): string | null | undefined {
  return container.querySelector('.theme-choice [aria-pressed="true"]')?.textContent;
}

beforeEach(() => {
  vi.clearAllMocks();
  saveSettingsMock.mockResolvedValue({ settings: {}, warning: null });
});

describe("Appearance", () => {
  it("offers Light, Dark and System with the stored theme pressed", () => {
    const container = render("system");

    const labels = [...container.querySelectorAll(".theme-choice button")].map((b) => b.textContent);
    expect(labels).toEqual(["Light", "Dark", "System"]);
    expect(pressedTheme(container)).toBe("System");
    expect(container.querySelector(".setting-row-hint")?.textContent).toBe(
      "Follows your system by default",
    );
  });

  it("switches this window at once, saves the choice and tells the other windows", async () => {
    const container = render("system");

    themeButton(container, "Dark").click();

    expect(applyAppThemeMock).toHaveBeenCalledWith("dark");
    await vi.waitFor(() => expect(emitSettingsChangedMock).toHaveBeenCalled());
    expect(saveSettingsMock).toHaveBeenCalledWith({ theme: "dark" });
  });

  it("goes back to the stored theme when saving fails", async () => {
    saveSettingsMock.mockRejectedValue(new Error("backend down"));
    const container = render("light");

    themeButton(container, "Dark").click();

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalled());
    expect(applyAppThemeMock).toHaveBeenLastCalledWith("light");
    expect(pressedTheme(container)).toBe("Light");
    expect(emitSettingsChangedMock).not.toHaveBeenCalled();
  });
});
