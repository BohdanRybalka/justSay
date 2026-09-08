import { describe, expect, it } from "vitest";
import { ApiAuthError, ApiRequestError } from "../api";
import { dictationErrorLabel, startErrorLabel } from "./error-label";

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

  it("names the 403 as somebody else's recording instead of inviting a retry", () => {
    const { label, toast } = dictationErrorLabel(new ApiRequestError("Recording is owned by another session", 403));

    expect(label).toBe("Recording is busy");
    expect(toast).toBe("Another window is using the microphone — stop it there and try again.");
  });

  it("reads the 403 by its status, not by what its body happens to say", () => {
    const { label } = dictationErrorLabel(
      new ApiRequestError("Recording is missing an owning session", 403),
    );

    expect(label).toBe("Recording is busy");
    expect(label).not.toBe("Add key in Settings");
  });

  it("leaves a refusal that is not a 403 on the generic label", () => {
    expect(dictationErrorLabel(new ApiRequestError("Not recording", 409)).label).toBe("Failed");
  });

  it("a non-Error rejection is stringified rather than crashing the handler", () => {
    expect(dictationErrorLabel("missing something").label).toBe("Add key in Settings");
    expect(dictationErrorLabel(undefined).label).toBe("Failed");
  });

});

describe("startErrorLabel", () => {
  it("keeps a refused start off the dictation wording, whatever the refusal says", () => {
    for (const failure of [
      new Error("Already recording"),
      new Error("Missing or invalid API token"),
      new Error("connection reset"),
    ]) {
      const { label, toast } = startErrorLabel(failure);

      expect(label).toBe("Start failed");
      expect(toast).toBe("Couldn't start recording — try again.");
    }
  });

  it("tells a 401 to restart the app instead of offering a retry that cannot work", () => {
    const { label, toast } = startErrorLabel(
      new ApiAuthError("Missing or invalid API token", { kind: "bridge-missing" }),
    );

    expect(label).toBe("Auth failed");
    expect(label).not.toBe("Start failed");
    expect(toast).toBe("JustSay could not authenticate to its own backend — restart the app.");
    expect(toast).not.toContain("try again");
    expect(toast).not.toContain("API key");
  });

});
