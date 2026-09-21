import { describe, expect, it } from "vitest";
import { nextTabAction } from "./tab-visibility";

const VISIBLE = false;
const HIDDEN = true;

describe("nextTabAction", () => {
  it("releases the tab when a visible window is dismissed", () => {
    expect(
      nextTabAction("hidden", VISIBLE),
      "any other action here leaves the tab polling for the life of the app",
    ).toBe("release");
  });

  it("ignores a second dismissal of a window already hidden", () => {
    expect(
      nextTabAction("hidden", HIDDEN),
      "a second release would run a tab's teardown twice on one dismissal",
    ).toBe("ignore");
  });

  it("resumes the tab when a hidden window is shown", () => {
    expect(
      nextTabAction("shown", HIDDEN),
      "any other action here leaves the window silent until the app restarts",
    ).toBe("resume");
  });

  it("ignores a show of a window that was never hidden", () => {
    expect(
      nextTabAction("shown", VISIBLE),
      "the tray item shows an already visible window, and a resume there would " +
        "start a second interval beside the live one",
    ).toBe("ignore");
  });
});
