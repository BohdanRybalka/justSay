import { describe, expect, it } from "vitest";
import { ApiAuthError, REQUEST_TIMEOUT_MS } from "../api";
import { TimedOutError } from "../timeout";
import { dictationErrorLabel } from "./error-label";

describe("dictationErrorLabel", () => {
  it("an ApiAuthError never renders the API-key label, even though its message contains 'missing'", () => {
    const error = new ApiAuthError("Missing or invalid API token", { kind: "bridge-missing" });

    const { label, toast } = dictationErrorLabel(error);

    expect(label).not.toBe("Add key in Settings");
    expect(toast).not.toContain("API key");
    expect(label).toBe("Auth failed");
  });

  it("a genuinely missing cloud key still routes the user to Settings", () => {
    const { label, toast } = dictationErrorLabel(new Error("Missing GEMINI_API_KEY"));

    expect(label).toBe("Add key in Settings");
    expect(toast).toBe("No API key set — add one in Settings.");
  });

  it("any other failure falls through to the generic label", () => {
    const { label, toast } = dictationErrorLabel(new Error("connection reset"));

    expect(label).toBe("Failed");
    expect(toast).toBe("Dictation failed — try again.");
  });

  it("a non-Error rejection is stringified rather than crashing the handler", () => {
    expect(dictationErrorLabel("missing something").label).toBe("Add key in Settings");
    expect(dictationErrorLabel(undefined).label).toBe("Failed");
  });

  it("a budget that expired names the wait and does not claim the dictation failed", () => {
    const { label, toast } = dictationErrorLabel(
      new TimedOutError(REQUEST_TIMEOUT_MS, "/pipeline/dictate"),
    );

    expect(label).not.toBe("Failed");
    expect(label).toBe("No answer");
    expect(toast).toContain("may not have been copied");
    expect(toast).not.toBe("Dictation failed — try again.");
  });
});
