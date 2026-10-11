// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { buildDownload, buildLocalStatus } from "../local-status-stub.test-helper";
import { drawRow, localModeRowHtml, localRow, localView, modeRowHtml } from "./mode-rows";

const MISSING = { model_downloaded: false };

describe("localRow — the download the row follows", () => {
  it("counts whole percent of the bytes that arrived", () => {
    const download = buildDownload({ stage: "downloading", bytes_done: 333, bytes_total: 1000 });

    expect(localRow(buildLocalStatus(MISSING), false, download)).toEqual({ state: "downloading", percent: 33 });
  });

  it("never passes 100 and has no percent before the size is known", () => {
    const over = buildDownload({ stage: "downloading", bytes_done: 1200, bytes_total: 1000 });
    const unknown = buildDownload({ stage: "downloading", bytes_done: 50, bytes_total: null });

    expect(localRow(buildLocalStatus(MISSING), false, over)).toEqual({ state: "downloading", percent: 100 });
    expect(localRow(buildLocalStatus(MISSING), false, unknown)).toEqual({ state: "downloading", percent: null });
  });

  it("reads a failed download as a failure of the download, with the backend's sentence", () => {
    const download = buildDownload({ stage: "failed", error: "The download was interrupted." });

    expect(localRow(buildLocalStatus(MISSING), false, download)).toEqual({
      state: "failed",
      reason: "The download was interrupted.",
      cause: "download",
    });
  });

  it("forgets a failed download once the model is on the disk", () => {
    const download = buildDownload({ stage: "failed", error: "The download was interrupted." });

    expect(localRow(buildLocalStatus(), false, download)).toEqual({ state: "installed" });
  });

  it("draws a download under way over the engine starting while Local is the mode", () => {
    const download = buildDownload({ stage: "downloading", bytes_total: 10 });

    expect(localRow(buildLocalStatus(MISSING), true, download).state).toBe("downloading");
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
    const view = localView({ state: "downloading", percent: 40 });

    expect(view).toMatchObject({ hint: "Installing…", locked: true, disabled: true, action: "" });
    expect(view.progress).toEqual({ percent: 40, label: "Downloading the model" });
  });

  it("offers Install with the size, and Try again with the reason", () => {
    expect(localView({ state: "not-installed", bytes: 1_624_555_275, tracked: true })).toMatchObject({
      hint: "Not installed · 1.5 GB",
      action: expect.stringContaining("Install"),
    });
    expect(localView({ state: "failed", reason: "No space.", cause: "download" })).toMatchObject({
      hint: "No space.",
      alert: true,
      action: expect.stringContaining("Try again"),
    });
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
    const view = localView({ state: "downloading", percent: 40 });

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

    drawRow(row, false, localView({ state: "downloading", percent: 40 }));
    drawRow(row, false, localView({ state: "downloading", percent: null }));

    expect(area.querySelector(".install-bar")!.hasAttribute("aria-valuenow")).toBe(false);
    expect(area.querySelector(".install-percent")!.textContent).toBe("");
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
