// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { checkMock, relaunchMock } = vi.hoisted(() => ({
  checkMock: vi.fn(),
  relaunchMock: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-updater", () => ({ check: checkMock }));
vi.mock("@tauri-apps/plugin-process", () => ({ relaunch: relaunchMock }));
vi.mock("@tauri-apps/api/app", () => ({ getVersion: vi.fn(async () => "0.13.1") }));

import { renderVersionRow } from "./version-row";

const LAST_UP_TO_DATE_KEY = "justsay.updates.lastUpToDateAt";

function renderRow(isDestroyed: () => boolean = () => false) {
  const row = document.createElement("div");
  renderVersionRow(row, isDestroyed);
  return {
    row,
    button: row.querySelector<HTMLButtonElement>(".version-action")!,
    status: row.querySelector<HTMLElement>(".version-status")!,
  };
}

function buildUpdate(downloadAndInstall = vi.fn(async () => {})) {
  return { version: "0.14.0", currentVersion: "0.13.1", downloadAndInstall };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Version row — what it says before any check", () => {
  it("shows the running version and says it never checked", async () => {
    const { row, button, status } = renderRow();

    await vi.waitFor(() => expect(row.querySelector(".version-number")!.textContent).toBe("0.13.1"));
    expect(status.textContent).toBe("Never checked for updates");
    expect(button.textContent).toBe("Check");
  });

  it("says when it last found the app up to date, from the stored time", () => {
    localStorage.setItem(LAST_UP_TO_DATE_KEY, String(Date.now() - 5 * 60 * 1000));

    expect(renderRow().status.textContent).toBe("Checked 5 minutes ago · up to date");
  });

  it("falls back to never checked when the stored time cannot be read", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    expect(renderRow().status.textContent).toBe("Never checked for updates");
  });
});

describe("Version row — check, install and restart", () => {
  it("a check that finds nothing says so, remembers when, and can check again", async () => {
    checkMock.mockResolvedValue(null);
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(status.textContent).toBe("Checked just now · up to date"));

    expect(Number(localStorage.getItem(LAST_UP_TO_DATE_KEY))).toBeGreaterThan(0);
    expect(button.disabled).toBe(false);
    expect(button.textContent).toBe("Check");
    button.click();
    await vi.waitFor(() => expect(checkMock).toHaveBeenCalledTimes(2));
  });

  it("names the version that is ready and installs it once, without checking again", async () => {
    const downloadAndInstall = vi.fn(async () => {});
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Install and restart"));
    expect(status.textContent).toBe("Version 0.14.0 is ready");

    button.click();
    await vi.waitFor(() => expect(relaunchMock).toHaveBeenCalledTimes(1));
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
    expect(checkMock).toHaveBeenCalledTimes(1);
  });

  it("keeps the button disabled for the whole install", async () => {
    let finishInstall!: () => void;
    const downloadAndInstall = vi.fn(() => new Promise<void>((resolve) => (finishInstall = resolve)));
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button } = renderRow();

    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Install and restart"));
    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Installing…"));
    expect(button.disabled).toBe(true);

    button.click();
    finishInstall();
    await vi.waitFor(() => expect(relaunchMock).toHaveBeenCalledTimes(1));
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
  });

  it("a failed install offers a retry that installs, not checks", async () => {
    const downloadAndInstall = vi
      .fn()
      .mockRejectedValueOnce(new Error("disk full"))
      .mockResolvedValueOnce(undefined);
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Install and restart"));
    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Try again"));
    expect(status.textContent).toBe("Install failed: disk full");
    expect(button.disabled).toBe(false);

    button.click();
    await vi.waitFor(() => expect(downloadAndInstall).toHaveBeenCalledTimes(2));
    expect(checkMock).toHaveBeenCalledTimes(1);
  });

  it("a relaunch failure does not report the finished install as failed", async () => {
    const downloadAndInstall = vi.fn(async () => {});
    checkMock.mockResolvedValue(buildUpdate(downloadAndInstall));
    relaunchMock.mockRejectedValue(new Error("process:allow-restart denied"));
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Install and restart"));
    button.click();
    await vi.waitFor(() => expect(button.textContent).toBe("Check"));

    expect(status.textContent).toContain("The update is installed");
    expect(status.textContent).not.toContain("Install failed");
    button.click();
    await vi.waitFor(() => expect(checkMock).toHaveBeenCalledTimes(2));
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
  });

  it("names an unpublished release manifest and lets the user check again", async () => {
    checkMock.mockRejectedValue(new Error("could not fetch a valid release json"));
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(status.textContent).toContain("not published yet"));

    expect(button.disabled).toBe(false);
    expect(button.textContent).toBe("Check");
  });

  it("a check that rejects with a non-Error still gives the button back", async () => {
    checkMock.mockRejectedValue(null);
    const { button, status } = renderRow();

    button.click();
    await vi.waitFor(() => expect(status.textContent).toBe("Check failed: null"));
    expect(button.disabled).toBe(false);
  });

  it("writes nothing once the panel is gone", async () => {
    let answer!: (update: null) => void;
    checkMock.mockReturnValue(new Promise((resolve) => (answer = resolve)));
    let destroyed = false;
    const { button, status } = renderRow(() => destroyed);

    button.click();
    await vi.waitFor(() => expect(checkMock).toHaveBeenCalled());
    destroyed = true;
    answer(null);
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(status.textContent).toBe("Checking for updates…");
    expect(localStorage.getItem(LAST_UP_TO_DATE_KEY)).toBeNull();
  });
});
