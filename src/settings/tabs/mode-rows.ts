/** The big radio rows that pick Cloud or Local model, what each says about itself, and the Local row's install state. */
import type { CloudKeyStatus, InstallStep, LocalSTTStatus, ModelDownload } from "../../api";
import { displayableError } from "../../notify";
import { computeIndicatorState } from "../../status-indicator";
import { icon } from "../../ui/icons";

export const CLOUD_KEY_MISSING = "Add a Groq key in Settings";
export const CLOUD_HINT = "Your API keys · fastest and most accurate";

const WHERE_IT_BECOMES_TEXT = {
  cloud: "in the cloud",
  local: "on this computer",
} as const;

/** Where `mode` turns speech into text, as a phrase that ends a sentence. */
export function whereItBecomesText(mode: "cloud" | "local"): string {
  return WHERE_IT_BECOMES_TEXT[mode];
}

/** The line on Meetings and Files that says where their audio becomes text and where that is chosen. */
export function modelNoteHtml(mode: "cloud" | "local"): string {
  return `<p class="panel-subtitle" id="model-note">Turned into text ${whereItBecomesText(mode)}. Change this on the Dictation page.</p>`;
}

/** Whether Cloud's only engine, Groq, has no key; an unread key status is not missing. */
export function cloudKeyMissing(cloud: CloudKeyStatus | null): boolean {
  return cloud !== null && !cloud.groq_key_set;
}

export type StepState = "pending" | "running" | "done" | "failed";

export interface RowProgress {
  /** Whole percent, or `null` while the size is not known yet. */
  percent: number | null;
  /** The step in hand. */
  label: string;
  /** One state per entry of `STEP_LABELS`. */
  steps: StepState[];
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
  | { state: "installing"; step: InstallStep; percent: number | null }
  | { state: "starting" }
  | { state: "installed" }
  | { state: "ready" }
  | { state: "failed"; reason: string; cause: "engine" }
  | { state: "failed"; reason: string; cause: "install"; step: InstallStep; percent: number | null };

const READ_FAILED = "Couldn't load this";

export const STEP_LABELS = [
  "Downloading the model",
  "Checking your computer",
  "Starting the engine",
  "Ready to use",
] as const;
const INSTALL_STEPS: readonly InstallStep[] = ["downloading", "checking", "starting"];
const ALL_STEPS_DONE = STEP_LABELS.length;

export function isInstalling(stage: ModelDownload["stage"]): stage is InstallStep {
  return (INSTALL_STEPS as readonly string[]).includes(stage);
}

/** Binary units, as the file manager shows the model file. */
export function formatModelSize(bytes: number): string {
  const gib = bytes / 2 ** 30;
  return gib >= 1 ? `${gib.toFixed(1)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;
}

function stepPercent(download: ModelDownload, step: InstallStep): number | null {
  if (step !== "downloading") return 100;
  if (!download.bytes_total) return null;
  return Math.min(100, Math.floor((download.bytes_done / download.bytes_total) * 100));
}

/** A failed download is over once the model is on the disk; a failure after it, once the engine runs. */
function failureStillApplies(read: LocalSTTStatus, step: InstallStep): boolean {
  return step === "downloading" ? !read.model_downloaded : !read.model_loaded;
}

/** The Local row for `read`. An engine failure counts only while Local is the
 *  mode (ADR 009); an engine this computer cannot run outranks it, and an install
 *  under way or failed outranks everything about an engine that is not there yet.
 *  `finishing` keeps the finished steps on show for the moment before the switch. */
export function localRow(
  read: LocalRead,
  selected: boolean,
  download: ModelDownload | null,
  finishing = false,
): LocalRow {
  if (read === "failed") return { state: "failed", reason: READ_FAILED, cause: "engine" };
  if (read === null) return { state: "checking" };
  if (!read.available) return { state: "unavailable" };
  const reported = read.model_download_reported ? download : null;
  if (reported && isInstalling(reported.stage)) {
    return { state: "installing", step: reported.stage, percent: stepPercent(reported, reported.stage) };
  }
  if (reported?.stage === "failed" && reported.error) {
    const step = reported.failed_step ?? "downloading";
    if (failureStillApplies(read, step)) {
      return { state: "failed", reason: reported.error, cause: "install", step, percent: stepPercent(reported, step) };
    }
  }
  const error = displayableError(read.last_error);
  const indicator = computeIndicatorState({ active: selected, ready: read.model_loaded, error });
  if (indicator === "error") return { state: "failed", reason: error!, cause: "engine" };
  const missing = !(read.package_installed && read.model_downloaded);
  const idleOrWaitingForADownload =
    indicator === "idle" || (indicator === "loading" && read.model_download_reported);
  if (idleOrWaitingForADownload && missing) {
    return { state: "not-installed", bytes: read.model_bytes, tracked: read.model_download_reported };
  }
  if (indicator === "loading") return { state: "starting" };
  return { state: finishing ? "ready" : "installed" };
}

function progressAt(reached: number, failed: boolean, percent: number | null): RowProgress {
  return {
    percent,
    label: STEP_LABELS[Math.min(reached, STEP_LABELS.length - 1)],
    steps: STEP_LABELS.map((_, i): StepState => {
      if (i < reached) return "done";
      if (i > reached) return "pending";
      return failed ? "failed" : "running";
    }),
  };
}

const INSTALLED_VIEW: RowView = {
  hint: "Installed · runs on this computer",
  alert: false,
  locked: false,
  disabled: false,
  action: '<span class="chip">Ready</span>',
};

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
    case "installing":
      return {
        hint: "Installing…",
        alert: false,
        locked: true,
        disabled: true,
        action: "",
        progress: progressAt(INSTALL_STEPS.indexOf(row.step), false, row.percent),
      };
    case "starting":
      return { hint: "Starting…", alert: false, locked: false, disabled: false, action: "" };
    case "installed":
      return INSTALLED_VIEW;
    case "ready":
      return { ...INSTALLED_VIEW, progress: progressAt(ALL_STEPS_DONE, false, 100) };
    case "failed":
      return {
        hint: row.reason,
        alert: true,
        locked: false,
        disabled: false,
        action: '<span class="btn btn-small">Try again</span>',
        progress: row.cause === "install" ? progressAt(INSTALL_STEPS.indexOf(row.step), true, row.percent) : null,
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
  const items = area.querySelectorAll<HTMLElement>(".install-step");
  progress.steps.forEach((state, i) => {
    const item = items[i];
    item.classList.toggle("install-step--running", state === "running");
    item.classList.toggle("install-step--done", state === "done");
    item.classList.toggle("install-step--failed", state === "failed");
    if (state === "running") item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
    item.querySelector("use")!.setAttribute("href", state === "failed" ? "#x" : "#check");
  });
}

export function drawRow(row: HTMLButtonElement, checked: boolean, view: RowView): void {
  row.setAttribute("aria-checked", String(checked));
  row.classList.toggle("mode-row--locked", view.locked);
  if (view.disabled) row.setAttribute("aria-disabled", "true");
  else row.removeAttribute("aria-disabled");
  const hint = row.querySelector<HTMLElement>(".mode-row-hint")!;
  hint.textContent = view.hint;
  hint.classList.toggle("mode-row-alert", view.alert);
  const action = row.querySelector<HTMLElement>(".mode-row-action")!;
  if (action.innerHTML !== view.action) action.innerHTML = view.action;
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

/** The Local model row with the progress area that opens under it while the model installs. */
export function localModeRowHtml(id: string): string {
  const steps = STEP_LABELS.map(
    (label) => `<li class="install-step"><span class="install-step-dot">${icon("check")}</span>${label}</li>`,
  ).join("");
  return `${modeRowHtml(id, "chip", "Local model")}
    <div class="mode-row-install" hidden>
      <div class="install-bar" role="progressbar" aria-label="Installing the model" aria-valuemin="0" aria-valuemax="100"><i></i></div>
      <div class="install-meta"><span class="install-label"></span><span class="install-percent"></span></div>
      <ol class="install-steps">${steps}</ol>
    </div>`;
}
