// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserSettings } from "../../api";

const {
  saveSettingsMock,
  emitSettingsChangedMock,
  applyAppThemeMock,
  notifyErrorMock,
  apiMock,
  invokeMock,
  renderKeysMock,
} = vi.hoisted(() => ({
  saveSettingsMock: vi.fn(),
  emitSettingsChangedMock: vi.fn(async () => {}),
  applyAppThemeMock: vi.fn(async () => {}),
  notifyErrorMock: vi.fn(async () => {}),
  apiMock: { getStorageInfo: vi.fn(), cleanupTemp: vi.fn(), clearHistory: vi.fn() },
  invokeMock: vi.fn(),
  renderKeysMock: vi.fn(),
}));

const CLOUD_STATUS = { gemini_key_set: false, groq_key_set: true };

vi.mock("../settings", () => ({
  saveSettings: saveSettingsMock,
  getCloudKeyStatus: () => CLOUD_STATUS,
}));
vi.mock("./dictation", () => ({ emitSettingsChanged: emitSettingsChangedMock }));
vi.mock("../../ui/theme", () => ({ applyAppTheme: applyAppThemeMock }));
vi.mock("../../notify", () => ({ notifyError: notifyErrorMock }));
vi.mock("../../api", () => ({ api: apiMock }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("./version-row", () => ({ renderVersionRow: vi.fn() }));
vi.mock("./keys", () => ({ renderKeys: renderKeysMock }));

import { renderSettingsPanel } from "./settings-panel";

function render(theme: UserSettings["theme"], initialPrompt = ""): HTMLElement {
  const container = document.createElement("div");
  document.body.replaceChildren(container);
  renderSettingsPanel(container, { theme, initial_prompt: initialPrompt } as UserSettings);
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
  apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 69_000_000 });
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

function buttonLabelled(root: HTMLElement, label: string): HTMLButtonElement {
  return [...root.querySelectorAll<HTMLButtonElement>("button")].find(
    (button) => button.textContent?.trim() === label,
  )!;
}

describe("Storage", () => {
  function storageRow(container: HTMLElement): HTMLElement {
    return container.querySelector<HTMLElement>(".storage-row")!;
  }

  it("says how much temporary audio there is", async () => {
    const row = storageRow(render("system"));

    await vi.waitFor(() =>
      expect(row.querySelector(".storage-size")!.textContent).toBe("65.8 MB of temporary audio"),
    );
  });

  it("opens the folder through the shell", async () => {
    const row = storageRow(render("system"));

    buttonLabelled(row, "Open folder").click();

    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith("open_scratch_folder"));
  });

  it("reports a folder that could not be opened", async () => {
    invokeMock.mockRejectedValue(new Error("no file manager"));
    const row = storageRow(render("system"));

    buttonLabelled(row, "Open folder").click();

    await vi.waitFor(() =>
      expect(notifyErrorMock).toHaveBeenCalledWith("Could not open the folder: no file manager"),
    );
  });

  it("asks once before clearing, and clears nothing on Cancel", async () => {
    const row = storageRow(render("system"));
    await vi.waitFor(() => expect(row.querySelector(".num")).not.toBeNull());

    buttonLabelled(row, "Clear").click();
    expect(row.querySelector(".storage-size")!.textContent).toBe("Clear 65.8 MB of temporary audio?");
    buttonLabelled(row, "Cancel").click();

    await vi.waitFor(() =>
      expect(row.querySelector(".storage-size")!.textContent).toBe("65.8 MB of temporary audio"),
    );
    expect(apiMock.cleanupTemp).not.toHaveBeenCalled();
    expect(buttonLabelled(row, "Open folder")).toBeDefined();
  });

  it("clears on the confirmation and shows the size that is left", async () => {
    apiMock.cleanupTemp.mockResolvedValue({ freed_bytes: 69_000_000 });
    const row = storageRow(render("system"));
    await vi.waitFor(() => expect(row.querySelector(".num")).not.toBeNull());
    apiMock.getStorageInfo.mockResolvedValue({ temp_size_bytes: 0 });

    buttonLabelled(row, "Clear").click();
    row.querySelector<HTMLButtonElement>(".btn-primary")!.click();

    await vi.waitFor(() =>
      expect(row.querySelector(".storage-size")!.textContent).toBe("0 B of temporary audio"),
    );
    expect(apiMock.cleanupTemp).toHaveBeenCalledOnce();
  });
});

describe("Delete all history", () => {
  function deleteArea(container: HTMLElement): HTMLElement {
    return container.querySelector<HTMLElement>(".history-delete")!;
  }

  it("asks before deleting, and deletes nothing on Cancel", async () => {
    const area = deleteArea(render("system"));

    buttonLabelled(area, "Delete all history").click();
    expect(area.textContent).toContain("Delete every recording? This can't be undone.");
    buttonLabelled(area, "Cancel").click();

    await vi.waitFor(() => expect(area.textContent).not.toContain("Delete every recording"));
    expect(apiMock.clearHistory).not.toHaveBeenCalled();
    expect(buttonLabelled(area, "Delete all history")).toBeDefined();
  });

  it("hands focus back to the button that asked once the question is answered", async () => {
    const area = deleteArea(render("system"));
    const start = buttonLabelled(area, "Delete all history");

    start.focus();
    start.click();
    buttonLabelled(area, "Cancel").click();

    await vi.waitFor(() => expect(document.activeElement).toBe(start));
  });

  it("deletes everything on the confirmation and says so", async () => {
    apiMock.clearHistory.mockResolvedValue({ deleted: 12 });
    const area = deleteArea(render("system"));

    buttonLabelled(area, "Delete all history").click();
    buttonLabelled(area, "Delete").click();

    await vi.waitFor(() =>
      expect(area.querySelector(".history-delete-status")!.textContent).toBe("History deleted"),
    );
    expect(apiMock.clearHistory).toHaveBeenCalledOnce();
  });

  it("says what went wrong when the backend refuses", async () => {
    apiMock.clearHistory.mockRejectedValue(new Error("database is locked"));
    const area = deleteArea(render("system"));

    buttonLabelled(area, "Delete all history").click();
    buttonLabelled(area, "Delete").click();

    await vi.waitFor(() =>
      expect(area.querySelector(".history-delete-status")!.textContent).toBe(
        "Could not delete: database is locked",
      ),
    );
    expect(buttonLabelled(area, "Delete all history").disabled).toBe(false);
  });
});

describe("the dictionary", () => {
  it("sits under the card, drawn from the stored words and saying where they are used", () => {
    const container = render("system", "Tauri, n8n");

    const dictionary = container.querySelector(".dictionary")!;
    expect(dictionary.previousElementSibling!.classList.contains("card")).toBe(true);
    expect(dictionary.querySelector(".group-label")!.textContent).toBe("YOUR DICTIONARY");
    expect([...dictionary.querySelectorAll(".word-chip")].map((chip) => chip.textContent)).toEqual([
      "Tauri",
      "n8n",
    ]);
    expect(dictionary.textContent).toContain("Used for dictation, meetings and files");
  });
});

describe("the dictionary's teardown", () => {
  it("goes with the panel, so a save that fails afterwards draws nothing", async () => {
    let refuse: (reason: Error) => void = () => {};
    saveSettingsMock.mockImplementationOnce(() => new Promise((_, reject) => (refuse = reject)));
    const container = document.createElement("div");
    document.body.replaceChildren(container);
    const panel = renderSettingsPanel(container, { theme: "system", initial_prompt: "" } as UserSettings);
    container.querySelector<HTMLInputElement>("#dictionary-input")!.value = "Tauri";
    container.querySelector<HTMLButtonElement>("#dictionary-add")!.click();
    await vi.waitFor(() => expect(saveSettingsMock).toHaveBeenCalledWith({ initial_prompt: "Tauri" }));

    panel.destroy();
    refuse(new Error("backend down"));

    await vi.waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith("backend down"));
    expect(container.querySelectorAll(".word-chip")).toHaveLength(1);
  });
});

describe("API keys", () => {
  it("fold away, closed, between the dictionary and Delete all history", () => {
    const container = render("system");

    const fold = container.querySelector<HTMLDetailsElement>("details.api-keys")!;
    expect(fold.open).toBe(false);
    expect(fold.classList.contains("fold")).toBe(true);
    expect(fold.querySelector("summary")!.textContent).toBe("API keys");
    expect(fold.previousElementSibling!.classList.contains("dictionary")).toBe(true);
    expect(fold.nextElementSibling!.classList.contains("history-delete")).toBe(true);
  });

  it("hold the key rows, drawn from the settings and the cloud key status", () => {
    const container = render("system");

    const rows = container.querySelector(".api-keys-rows");
    expect(renderKeysMock).toHaveBeenCalledWith(rows, { theme: "system", initial_prompt: "" }, CLOUD_STATUS);
  });
});
