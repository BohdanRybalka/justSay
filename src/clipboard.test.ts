import { afterEach, describe, expect, it, vi } from "vitest";

const { invokeMock } = vi.hoisted(() => ({ invokeMock: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const { copyToClipboard, deliverDictation } = await import("./clipboard");

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

describe("deliverDictation", () => {
  it("only copies when pasting is off", async () => {
    invokeMock.mockResolvedValue(undefined);

    await expect(deliverDictation("привіт", false)).resolves.toBe("copied");

    expect(invokeMock).toHaveBeenCalledExactlyOnceWith("write_clipboard_text", { text: "привіт" });
  });

  it("reads a refused copy as failed when pasting is off", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    invokeMock.mockRejectedValue("the clipboard stayed busy");

    await expect(deliverDictation("text", false)).resolves.toBe("failed");
  });

  it.each(["pasted", "copied", "failed"] as const)(
    "hands the text to the paste command and reports its answer, %s",
    async (answer) => {
      invokeMock.mockResolvedValue(answer);

      await expect(deliverDictation("привіт", true)).resolves.toBe(answer);

      expect(invokeMock).toHaveBeenCalledExactlyOnceWith("paste_text", { text: "привіт" });
    },
  );

  it("reads a paste command that never answers as failed, after the key wait", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    invokeMock.mockImplementation(() => new Promise(() => {}));
    let settled = false;

    const delivered = deliverDictation("text", true).finally(() => (settled = true));
    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledOnce());
    await vi.advanceTimersByTimeAsync(3000);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(2000);

    await expect(delivered).resolves.toBe("failed");
  });
});
