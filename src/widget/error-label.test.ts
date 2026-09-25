import { describe, expect, it } from "vitest";
import { ApiAuthError, ApiRequestError, CONFIGURATION_ERROR_CODE } from "../api";
import { TimedOutError } from "../timeout";
import { dictationErrorLabel, startErrorLabel } from "./error-label";

const PILL_LABELS = [
  "No connection",
  "Add an API key",
  "Mic is busy",
  "Didn't work",
];

const authFailure = () =>
  new ApiAuthError("Missing or invalid API token", { kind: "bridge-missing" });

describe("dictationErrorLabel", () => {
  it.each([
    ["a 401 from the app's own backend", authFailure(), "No connection"],
    ["a refused connection", new TypeError("Failed to fetch"), "No connection"],
    ["a request that ran out of its budget", new TimedOutError(15_000, "/pipeline/dictate"), "No connection"],
    ["a missing cloud key", new ApiRequestError("any text", 400, CONFIGURATION_ERROR_CODE), "Add an API key"],
    ["another window holding the microphone", new ApiRequestError("owned elsewhere", 403), "Mic is busy"],
    ["a crash", new ApiRequestError("boom", 500), "Didn't work"],
    ["a 409", new ApiRequestError("Not recording", 409), "Didn't work"],
    ["a plain error", new Error("connection reset"), "Didn't work"],
    ["a thrown string", "missing something", "Didn't work"],
    ["nothing at all", undefined, "Didn't work"],
  ])("labels %s as %s", (_name, error, expected) => {
    expect(dictationErrorLabel(error).label).toBe(expected);
  });

  it("chooses the key label by the refusal's code, never by what the body says", () => {
    const crashAboutKeys = new ApiRequestError(
      "Gemini API key is missing. Go to Settings → Keys and add your key.",
      500,
      null,
    );
    const authMentioningMissing = authFailure();

    expect(dictationErrorLabel(crashAboutKeys).label).toBe("Didn't work");
    expect(dictationErrorLabel(authMentioningMissing).label).not.toBe("Add an API key");
    expect(dictationErrorLabel(authMentioningMissing).toast).not.toContain("API key");
  });

  it("keeps the full sentence for the notification", () => {
    expect(dictationErrorLabel(new ApiRequestError("x", 400, CONFIGURATION_ERROR_CODE)).toast).toBe(
      "No API key set — add one in Settings.",
    );
    expect(dictationErrorLabel(new ApiRequestError("x", 403)).toast).toBe(
      "Another window is using the microphone — stop it there and try again.",
    );
    expect(dictationErrorLabel(new Error("x")).toast).toBe("Dictation failed — try again.");
  });
});

describe("startErrorLabel", () => {
  it.each([
    ["a 401 from the app's own backend", authFailure(), "No connection"],
    ["a refused connection", new TypeError("Failed to fetch"), "No connection"],
    ["a recorder already held", new ApiRequestError("Already recording", 409), "Mic is busy"],
    ["a refusal whose text says missing", new Error("Missing or invalid API token"), "Didn't work"],
    ["a missing key, which a start never needs", new ApiRequestError("x", 400, CONFIGURATION_ERROR_CODE), "Didn't work"],
  ])("labels %s as %s", (_name, error, expected) => {
    expect(startErrorLabel(error).label).toBe(expected);
  });

  it("tells a 401 to restart the app instead of offering a retry that cannot work", () => {
    const { toast } = startErrorLabel(authFailure());

    expect(toast).toBe("JustSay could not authenticate to its own backend — restart the app.");
  });

  it("says the recording never began for anything else", () => {
    expect(startErrorLabel(new Error("x")).toast).toBe("Couldn't start recording — try again.");
  });
});

describe("the pill's failure labels", () => {
  it("come only from the short list the user approved", () => {
    const failures = [
      authFailure(),
      new TypeError("Failed to fetch"),
      new ApiRequestError("x", 400, CONFIGURATION_ERROR_CODE),
      new ApiRequestError("x", 403),
      new ApiRequestError("x", 409),
      new ApiRequestError("x", 500),
      new Error("x"),
    ];

    for (const failure of failures) {
      expect(PILL_LABELS).toContain(dictationErrorLabel(failure).label);
      expect(PILL_LABELS).toContain(startErrorLabel(failure).label);
    }
  });
});
