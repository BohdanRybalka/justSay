import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  confirm: vi.fn(),
}));

vi.mock("../api", () => ({
  api: {
    getHistory: vi.fn(),
    clearHistory: vi.fn(),
  },
}));

const { formatEntryCount } = await import("./history-list");

const TRANSCRIPTS = { singular: "transcript", plural: "transcripts" };
const ENTRIES = { singular: "entry", plural: "entries" };

describe("formatEntryCount — the two tabs keep their own wording", () => {
  it("uses the plural for zero", () => {
    expect(formatEntryCount(0, TRANSCRIPTS)).toBe("0 transcripts");
    expect(formatEntryCount(0, ENTRIES)).toBe("0 entries");
  });

  it("uses the singular for exactly one", () => {
    expect(formatEntryCount(1, TRANSCRIPTS)).toBe("1 transcript");
    expect(formatEntryCount(1, ENTRIES)).toBe("1 entry");
  });

  it("uses the plural for two", () => {
    expect(formatEntryCount(2, TRANSCRIPTS)).toBe("2 transcripts");
    expect(formatEntryCount(2, ENTRIES)).toBe("2 entries");
  });
});
