import { afterEach, describe, expect, it, vi } from "vitest";

const { invokeMock } = vi.hoisted(() => ({ invokeMock: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const { copyToClipboard } = await import("./clipboard");

afterEach(() => {
  vi.useRealTimers();
  invokeMock.mockReset();
  vi.restoreAllMocks();
});

describe("copyToClipboard", () => {
  it("hands the text to the shell's clipboard command", async () => {
    invokeMock.mockResolvedValue(undefined);

    await expect(copyToClipboard("привіт світ")).resolves.toBe(true);

    expect(invokeMock).toHaveBeenCalledExactlyOnceWith("write_clipboard_text", {
      text: "привіт світ",
    });
  });

  it("answers false when the shell refuses the write", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    invokeMock.mockRejectedValue("the clipboard stayed busy");

    await expect(copyToClipboard("text")).resolves.toBe(false);
  });

  it("answers false when the shell never answers", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    invokeMock.mockImplementation(() => new Promise(() => {}));

    const copied = copyToClipboard("text");
    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledOnce());
    await vi.advanceTimersByTimeAsync(3000);

    await expect(copied).resolves.toBe(false);
  });
});
