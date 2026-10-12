// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { buildDownload, buildLocalStatus } from "../local-status-stub.test-helper";
import { drawRow, localModeRowHtml, localRow, localView, modeRowHtml } from "./mode-rows";

const MISSING = { model_downloaded: false };

describe("localRow — the install the row follows", () => {
  it("counts whole percent of the bytes that arrived", () => {
    const download = buildDownload({ stage: "downloading", bytes_done: 333, bytes_total: 1000 });

    expect(localRow(buildLocalStatus(MISSING), false, download)).toEqual({
      state: "installing",
      step: "downloading",
      percent: 33,
    });
  });

  it("never passes 100 and has no percent before the size is known", () => {
    const over = buildDownload({ stage: "downloading", bytes_done: 1200, bytes_total: 1000 });
    const unknown = buildDownload({ stage: "downloading", bytes_done: 50, bytes_total: null });

    expect(localRow(buildLocalStatus(MISSING), false, over)).toMatchObject({ percent: 100 });
    expect(localRow(buildLocalStatus(MISSING), false, unknown)).toMatchObject({ percent: null });
  });

  it("follows the steps after the download with the model already on the disk", () => {
    const checking = buildDownload({ stage: "checking", bytes_done: 1000, bytes_total: 1000 });
    const starting = buildDownload({ stage: "starting" });

    expect(localRow(buildLocalStatus(), false, checking)).toEqual({
      state: "installing",
      step: "checking",
      percent: 100,
    });
    expect(localRow(buildLocalStatus(), false, starting)).toEqual({
      state: "installing",
      step: "starting",
      percent: 100,
    });
  });

  it("keeps the steps on show while the engine is up but the install has not said it is done", () => {
    const starting = buildDownload({ stage: "starting" });

    expect(localRow(buildLocalStatus({ model_loaded: true }), false, starting).state).toBe("installing");
  });

  it("reads a failed download as a failure of the install at its step, with the backend's sentence", () => {
    const download = buildDownload({
      stage: "failed",
      error: "The download was interrupted.",
      failed_step: "downloading",
      bytes_done: 400,
      bytes_total: 1000,
    });

    expect(localRow(buildLocalStatus(MISSING), false, download)).toEqual({
      state: "failed",
      reason: "The download was interrupted.",
      cause: "install",
      step: "downloading",
      percent: 40,
    });
  });

  it("reads a failure after the download as the install's too, at 100%", () => {
    const download = buildDownload({ stage: "failed", error: "The engine did not start.", failed_step: "starting" });

    expect(localRow(buildLocalStatus(), false, download)).toEqual({
      state: "failed",
      reason: "The engine did not start.",
      cause: "install",
      step: "starting",
      percent: 100,
    });
  });

  it("forgets a failed download once the model is on the disk", () => {
    const download = buildDownload({ stage: "failed", error: "The download was interrupted.", failed_step: "downloading" });

    expect(localRow(buildLocalStatus(), false, download)).toEqual({ state: "installed" });
  });

  it("forgets a failure to start the engine once the engine runs", () => {
    const download = buildDownload({ stage: "failed", error: "The engine did not start.", failed_step: "starting" });

    expect(localRow(buildLocalStatus({ model_loaded: true }), false, download)).toEqual({ state: "installed" });
  });

  it("keeps the finished steps on show for the pause before the switch, and not after it", () => {
    const status = buildLocalStatus({ model_loaded: true });
    const done = buildDownload({ stage: "done" });

    expect(localRow(status, false, done, true)).toEqual({ state: "ready" });
    expect(localRow(status, false, done, false)).toEqual({ state: "installed" });
  });

  it("draws an install under way over the engine starting while Local is the mode", () => {
    const download = buildDownload({ stage: "downloading", bytes_total: 10 });

    expect(localRow(buildLocalStatus(MISSING), true, download).state).toBe("installing");
    expect(localRow(buildLocalStatus(), true, null).state).toBe("starting");
  });

  it("ignores a download state for an engine that does not report one", () => {
    const status = buildLocalStatus({ ...MISSING, model_download_reported: false });
    const download = buildDownload({ stage: "downloading", bytes_total: 10 });

    expect(localRow(status, false, download)).toEqual({
      state: "not-installed",
      bytes: status.model_bytes,
      tracked: false,
    });
  });

  it("offers Install for a missing model while Local is the mode and nothing downloads yet", () => {
    expect(localRow(buildLocalStatus(MISSING), true, buildDownload())).toEqual({
      state: "not-installed",
      bytes: 1_624_555_275,
      tracked: true,
    });
  });

  it("still says Starting for an engine that reports no download while Local is the mode", () => {
    const status = buildLocalStatus({ ...MISSING, model_download_reported: false });

    expect(localRow(status, true, null).state).toBe("starting");
  });

  it("lets an engine this computer cannot run outrank a download", () => {
    const download = buildDownload({ stage: "downloading", bytes_total: 10 });

    expect(localRow(buildLocalStatus({ ...MISSING, available: false }), false, download).state).toBe("unavailable");
  });

  it("marks a missing model on an engine that reports downloads as one Install can fetch", () => {
    expect(localRow(buildLocalStatus(MISSING), false, buildDownload())).toEqual({
      state: "not-installed",
      bytes: 1_624_555_275,
      tracked: true,
    });
  });
});

describe("localView", () => {
  it("opens the progress step and closes the row while the model downloads", () => {
    const view = localView({ state: "installing", step: "downloading", percent: 40 });

    expect(view).toMatchObject({ hint: "Installing…", locked: true, disabled: true, action: "" });
    expect(view.progress).toEqual({
      percent: 40,
      label: "Downloading the model",
      steps: ["running", "pending", "pending", "pending"],
    });
  });

  it("checks off the steps behind the one in hand", () => {
    expect(localView({ state: "installing", step: "checking", percent: 100 }).progress).toEqual({
      percent: 100,
      label: "Checking your computer",
      steps: ["done", "running", "pending", "pending"],
    });
    expect(localView({ state: "installing", step: "starting", percent: 100 }).progress).toEqual({
      percent: 100,
      label: "Starting the engine",
      steps: ["done", "done", "running", "pending"],
    });
  });

  it("says Installed with a Ready chip and all four steps done in the pause before the switch", () => {
    const view = localView({ state: "ready" });

    expect(view).toMatchObject({
      hint: "Installed · runs on this computer",
      action: expect.stringContaining("Ready"),
      locked: false,
    });
    expect(view.progress).toEqual({
      percent: 100,
      label: "Ready to use",
      steps: ["done", "done", "done", "done"],
    });
  });

  it("offers Install with the size, and Try again with the reason", () => {
    expect(localView({ state: "not-installed", bytes: 1_624_555_275, tracked: true })).toMatchObject({
      hint: "Not installed · 1.5 GB",
      action: expect.stringContaining("Install"),
    });
    expect(localView({ state: "failed", reason: "No space.", cause: "engine" })).toMatchObject({
      hint: "No space.",
      alert: true,
      action: expect.stringContaining("Try again"),
    });
  });

  it("marks the step a failed install stopped in, and shows no steps for an engine failure", () => {
    const failed = localView({ state: "failed", reason: "No engine.", cause: "install", step: "checking", percent: 100 });

    expect(failed.progress?.steps).toEqual(["done", "failed", "pending", "pending"]);
    expect(localView({ state: "failed", reason: "Out of memory.", cause: "engine" }).progress).toBeNull();
  });
});

describe("drawRow — the progress area under the Local row", () => {
  function mount() {
    const host = document.createElement("div");
    host.innerHTML = `${modeRowHtml("cloud", "cloud", "Cloud")}${localModeRowHtml("local")}`;
    const row = host.querySelector<HTMLButtonElement>("#local")!;
    const area = host.querySelector<HTMLElement>(".mode-row-install")!;
    return { host, row, area };
  }

  it("shows the bar, the percentage and the label, then hides them again", () => {
    const { row, area } = mount();
    const view = localView({ state: "installing", step: "downloading", percent: 40 });

    drawRow(row, false, view);

    expect(area.hidden).toBe(false);
    expect(area.querySelector<HTMLElement>(".install-bar>i")!.style.width).toBe("40%");
    expect(area.querySelector(".install-bar")!.getAttribute("aria-valuenow")).toBe("40");
    expect(area.querySelector(".install-percent")!.textContent).toBe("40%");
    expect(area.querySelector(".install-label")!.textContent).toBe("Downloading the model");

    drawRow(row, false, localView({ state: "installed" }));
    expect(area.hidden).toBe(true);
  });

  it("leaves the bar without a value while the size is unknown", () => {
    const { row, area } = mount();

    drawRow(row, false, localView({ state: "installing", step: "downloading", percent: 40 }));
    drawRow(row, false, localView({ state: "installing", step: "downloading", percent: null }));

    expect(area.querySelector(".install-bar")!.hasAttribute("aria-valuenow")).toBe(false);
    expect(area.querySelector(".install-percent")!.textContent).toBe("");
  });

  it("draws the four steps as spinner, check or failure marker as they stand", () => {
    const { row, area } = mount();
    const items = Array.from(area.querySelectorAll<HTMLElement>(".install-step"));
    const marker = (i: number) => items[i].querySelector("use")!.getAttribute("href");

    expect(items.map((item) => item.textContent)).toEqual([
      "Downloading the model",
      "Checking your computer",
      "Starting the engine",
      "Ready to use",
    ]);

    drawRow(row, false, localView({ state: "installing", step: "starting", percent: 100 }));
    expect(items.map((item) => item.className)).toEqual([
      "install-step install-step--done",
      "install-step install-step--done",
      "install-step install-step--running",
      "install-step",
    ]);
    expect(items[2].getAttribute("aria-current")).toBe("step");
    expect(items[0].hasAttribute("aria-current")).toBe(false);

    drawRow(row, false, localView({ state: "failed", reason: "x", cause: "install", step: "starting", percent: 100 }));
    expect(items[2].className).toBe("install-step install-step--failed");
    expect(marker(2)).toBe("#x");
    expect(marker(0)).toBe("#check");
    expect(items[2].hasAttribute("aria-current")).toBe(false);

    drawRow(row, false, localView({ state: "ready" }));
    expect(items.every((item) => item.classList.contains("install-step--done"))).toBe(true);
    expect(marker(2)).toBe("#check");
  });

  it("leaves the action buttons in place while a redraw changes nothing about them", () => {
    const { row } = mount();
    const view = localView({ state: "not-installed", bytes: 100, tracked: true });
    drawRow(row, false, view);
    const button = row.querySelector(".btn");

    drawRow(row, false, { ...view, hint: "Not installed · 1.5 GB" });

    expect(row.querySelector(".btn")).toBe(button);
  });

  it("draws the Cloud row, which has no progress area, without one", () => {
    const { host } = mount();
    const cloud = host.querySelector<HTMLButtonElement>("#cloud")!;

    expect(() => drawRow(cloud, true, localView({ state: "installed" }))).not.toThrow();
    expect(host.querySelectorAll(".mode-row-install")).toHaveLength(1);
  });
});
