import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();
const relaunchMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/plugin-process", () => ({ relaunch: relaunchMock }));

beforeEach(() => {
  invokeMock.mockReset();
  relaunchMock.mockReset();
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("readBackendFailure", () => {
  it("is the reason the shell recorded", async () => {
    const { readBackendFailure } = await import("./backend-down");
    invokeMock.mockResolvedValue("Backend spawn failed: boom");

    await expect(readBackendFailure()).resolves.toBe("Backend spawn failed: boom");
    expect(invokeMock).toHaveBeenCalledWith("backend_failure_reason");
  });

  it("is null when the shell has none", async () => {
    const { readBackendFailure } = await import("./backend-down");
    invokeMock.mockResolvedValue(null);

    await expect(readBackendFailure()).resolves.toBeNull();
  });

  it("is null rather than a rejection when the shell refuses", async () => {
    const { readBackendFailure } = await import("./backend-down");
    invokeMock.mockRejectedValue(new Error("command backend_failure_reason not found"));

    await expect(readBackendFailure()).resolves.toBeNull();
  });

  it("is null once the shell has not answered in time, so the screen is never held on it", async () => {
    vi.useFakeTimers();
    const { readBackendFailure } = await import("./backend-down");
    invokeMock.mockReturnValue(new Promise(() => {}));

    const read = readBackendFailure();
    await vi.advanceTimersByTimeAsync(3000);

    await expect(read).resolves.toBeNull();
  });
});

describe("restartJustSay", () => {
  it("relaunches the app", async () => {
    const { restartJustSay } = await import("./backend-down");
    relaunchMock.mockResolvedValue(undefined);

    await restartJustSay();

    expect(relaunchMock).toHaveBeenCalledTimes(1);
  });
});
