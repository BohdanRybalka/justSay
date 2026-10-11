/** The big radio rows that pick Cloud or Local model, what each says about itself, and the Local row's install state. */
import type { CloudKeyStatus, LocalSTTStatus, ModelDownload } from "../../api";
import { displayableError } from "../../notify";
import { computeIndicatorState } from "../../status-indicator";
import { icon } from "../../ui/icons";

export const CLOUD_KEY_MISSING = "Add a Groq key in Settings";
export const CLOUD_HINT = "Your API keys · fastest and most accurate";

/** Whether Cloud's only engine, Groq, has no key; an unread key status is not missing. */
export function cloudKeyMissing(cloud: CloudKeyStatus | null): boolean {
  return cloud !== null && !cloud.groq_key_set;
}

export interface RowProgress {
  /** Whole percent, or `null` while the size is not known yet. */
  percent: number | null;
  label: string;
}

export interface RowView {
  hint: string;
  alert: boolean;
  locked: boolean;
  disabled: boolean;
  action: string;
  progress?: RowProgress | null;
}

/** The last status read: `null` before the first answer, `"failed"` when the read got none. */
export type LocalRead = LocalSTTStatus | "failed" | null;

export type LocalRow =
  | { state: "checking" }
  | { state: "unavailable" }
  | { state: "not-installed"; bytes: number | null; tracked: boolean }
  | { state: "downloading"; percent: number | null }
  | { state: "starting" }
  | { state: "installed" }
  | { state: "failed"; reason: string; cause: "download" | "engine" };

const READ_FAILED = "Couldn't load this";

/** Binary units, as the file manager shows the model file. */
export function formatModelSize(bytes: number): string {
  const gib = bytes / 2 ** 30;
  return gib >= 1 ? `${gib.toFixed(1)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;
}

function downloadPercent(download: ModelDownload): number | null {
  if (!download.bytes_total) return null;
  return Math.min(100, Math.floor((download.bytes_done / download.bytes_total) * 100));
}

/** The Local row for `read`. An engine failure counts only while Local is the
 *  mode (ADR 009); an engine this computer cannot run outranks it, and a download
 *  under way or failed outranks everything about an engine that is not there yet. */
export function localRow(read: LocalRead, selected: boolean, download: ModelDownload | null): LocalRow {
  if (read === "failed") return { state: "failed", reason: READ_FAILED, cause: "engine" };
  if (read === null) return { state: "checking" };
  if (!read.available) return { state: "unavailable" };
  const pending = read.model_download_reported && !read.model_downloaded ? download : null;
  if (pending?.stage === "downloading") return { state: "downloading", percent: downloadPercent(pending) };
  if (pending?.stage === "failed" && pending.error) {
    return { state: "failed", reason: pending.error, cause: "download" };
  }
  const error = displayableError(read.last_error);
  const indicator = computeIndicatorState({ active: selected, ready: read.model_loaded, error });
  if (indicator === "error") return { state: "failed", reason: error!, cause: "engine" };
  if (indicator === "loading") return { state: "starting" };
  if (indicator === "idle" && !(read.package_installed && read.model_downloaded)) {
    return { state: "not-installed", bytes: read.model_bytes, tracked: read.model_download_reported };
  }
  return { state: "installed" };
}

export function localView(row: LocalRow): RowView {
  switch (row.state) {
    case "checking":
      return { hint: "…", alert: false, locked: false, disabled: false, action: "" };
    case "unavailable":
      return { hint: "Not available on this computer", alert: false, locked: true, disabled: true, action: "" };
    case "not-installed":
      return {
        hint: row.bytes === null ? "Not installed" : `Not installed · ${formatModelSize(row.bytes)}`,
        alert: false,
        locked: true,
        disabled: false,
        action: '<span class="btn btn-blue btn-small">Install</span>',
      };
    case "downloading":
      return {
        hint: "Installing…",
        alert: false,
        locked: true,
        disabled: true,
        action: "",
        progress: { percent: row.percent, label: "Downloading the model" },
      };
    case "starting":
      return { hint: "Starting…", alert: false, locked: false, disabled: false, action: "" };
    case "installed":
      return {
        hint: "Installed · runs on this computer",
        alert: false,
        locked: false,
        disabled: false,
        action: '<span class="chip">Ready</span>',
      };
    case "failed":
      return {
        hint: row.reason,
        alert: true,
        locked: false,
        disabled: false,
        action: '<span class="btn btn-small">Try again</span>',
      };
  }
}

function drawProgress(area: HTMLElement, progress: RowProgress | null): void {
  area.hidden = progress === null;
  if (progress === null) return;
  const bar = area.querySelector<HTMLElement>(".install-bar")!;
  if (progress.percent === null) bar.removeAttribute("aria-valuenow");
  else bar.setAttribute("aria-valuenow", String(progress.percent));
  area.querySelector<HTMLElement>(".install-bar>i")!.style.width = `${progress.percent ?? 0}%`;
  area.querySelector<HTMLElement>(".install-label")!.textContent = progress.label;
  area.querySelector<HTMLElement>(".install-percent")!.textContent =
    progress.percent === null ? "" : `${progress.percent}%`;
}

export function drawRow(row: HTMLButtonElement, checked: boolean, view: RowView): void {
  row.setAttribute("aria-checked", String(checked));
  row.classList.toggle("mode-row--locked", view.locked);
  if (view.disabled) row.setAttribute("aria-disabled", "true");
  else row.removeAttribute("aria-disabled");
  const hint = row.querySelector<HTMLElement>(".mode-row-hint")!;
  hint.textContent = view.hint;
  hint.classList.toggle("mode-row-alert", view.alert);
  row.querySelector<HTMLElement>(".mode-row-action")!.innerHTML = view.action;
  const area = row.nextElementSibling;
  if (area?.classList.contains("mode-row-install")) drawProgress(area as HTMLElement, view.progress ?? null);
}

export function modeRowHtml(id: string, iconName: "cloud" | "chip", title: string): string {
  return `
    <button type="button" class="mode-row" id="${id}" role="radio" aria-checked="false">
      <span class="mode-row-radio"><i></i></span>
      <span class="mode-row-icon">${icon(iconName, "large")}</span>
      <span class="mode-row-text"><b>${title}</b><small class="mode-row-hint"></small></span>
      <span class="setting-row-controls mode-row-action"></span>
    </button>`;
}

/** The Local model row with the progress area that opens under it while the model downloads. */
export function localModeRowHtml(id: string): string {
  return `${modeRowHtml(id, "chip", "Local model")}
    <div class="mode-row-install" hidden>
      <div class="install-bar" role="progressbar" aria-label="Downloading the model" aria-valuemin="0" aria-valuemax="100"><i></i></div>
      <div class="install-meta"><span class="install-label"></span><span class="install-percent"></span></div>
    </div>`;
}
