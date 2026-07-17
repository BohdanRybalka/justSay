import { describe, expect, it } from "vitest";
import { computeDoneStatus } from "./done-status";

describe("computeDoneStatus", () => {
  it("non-empty transcript + copied to clipboard: Copied label", () => {
    expect(
      computeDoneStatus({ text: "hello world", duration_ms: 1234, copied_to_clipboard: true }),
    ).toEqual({ label: "Copied", elapsedSeconds: 1.234 });
  });

  it("non-empty transcript + clipboard copy failed: Copy failed label", () => {
    expect(
      computeDoneStatus({ text: "hello world", duration_ms: 4100, copied_to_clipboard: false }),
    ).toEqual({ label: "Copy failed", elapsedSeconds: 4.1 });
  });

  it("whitespace-only text: null (nothing to show)", () => {
    expect(computeDoneStatus({ text: "   ", duration_ms: 500, copied_to_clipboard: true })).toBeNull();
    expect(computeDoneStatus({ text: "", duration_ms: 500, copied_to_clipboard: true })).toBeNull();
  });
});
