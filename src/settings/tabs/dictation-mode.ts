/**
 * WHAT TURNS IT INTO TEXT: the Cloud and Local model radio rows. Cloud says
 * whether the Groq key is set; Local draws the local engine's state,
 * read every 3 s while the window is shown. Picking a row switches the mode.
 */
import { api, type CloudKeyStatus, type LocalSTTStatus, type UserSettings } from "../../api";
import { getCloudKeyStatus, loadSettings, type TabLifecycle } from "../settings";
import { displayableError, notifyError } from "../../notify";
import { isStaleStatusResponse } from "../../stale-response";
import { computeIndicatorState, onIndicatorStateChange } from "../../status-indicator";
import { icon } from "../../ui/icons";

type Mode = UserSettings["stt_mode"];

/** The last status read: `null` before the first answer, `"failed"` when the read got none. */
export type LocalRead = LocalSTTStatus | "failed" | null;

export type LocalRow =
  | { state: "checking" }
  | { state: "unavailable" }
  | { state: "not-installed"; bytes: number | null }
  | { state: "starting" }
  | { state: "installed" }
  | { state: "failed"; reason: string };

const CLOUD_HINT = "Your API keys · fastest and most accurate";
const CLOUD_KEY_MISSING = "Add a Groq key in Settings";
const READ_FAILED = "Couldn't load this";

/** Whether Cloud's only engine, Groq, has no key; an unread key status is not missing. */
export function cloudKeyMissing(cloud: CloudKeyStatus | null): boolean {
  return cloud !== null && !cloud.groq_key_set;
}

/** Binary units, as the file manager shows the model file. */
export function formatModelSize(bytes: number): string {
  const gib = bytes / 2 ** 30;
  return gib >= 1 ? `${gib.toFixed(1)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;
}

/** The Local row for `read`. An engine failure counts only while Local is the
 *  mode (ADR 009); an engine this computer cannot run outranks it. */
export function localRow(read: LocalRead, selected: boolean): LocalRow {
  if (read === "failed") return { state: "failed", reason: READ_FAILED };
  if (read === null) return { state: "checking" };
  if (!read.available) return { state: "unavailable" };
  const error = displayableError(read.last_error);
  const indicator = computeIndicatorState({ active: selected, ready: read.model_loaded, error });
  if (indicator === "error") return { state: "failed", reason: error! };
  if (indicator === "loading") return { state: "starting" };
  if (indicator === "idle" && !(read.package_installed && read.model_downloaded)) {
    return { state: "not-installed", bytes: read.model_bytes };
  }
  return { state: "installed" };
}

interface RowView {
  hint: string;
  alert: boolean;
  locked: boolean;
  disabled: boolean;
  action: string;
}

function localView(row: LocalRow): RowView {
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

function drawRow(row: HTMLButtonElement, checked: boolean, view: RowView): void {
  row.setAttribute("aria-checked", String(checked));
  row.classList.toggle("mode-row--locked", view.locked);
  if (view.disabled) row.setAttribute("aria-disabled", "true");
  else row.removeAttribute("aria-disabled");
  const hint = row.querySelector<HTMLElement>(".mode-row-hint")!;
  hint.textContent = view.hint;
  hint.classList.toggle("mode-row-alert", view.alert);
  row.querySelector<HTMLElement>(".mode-row-action")!.innerHTML = view.action;
}

function modeRowHtml(id: string, iconName: "cloud" | "chip", title: string): string {
  return `
    <button type="button" class="mode-row" id="${id}" role="radio" aria-checked="false">
      <span class="mode-row-radio"><i></i></span>
      <span class="mode-row-icon">${icon(iconName, "large")}</span>
      <span class="mode-row-text"><b>${title}</b><small class="mode-row-hint"></small></span>
      <span class="setting-row-controls mode-row-action"></span>
    </button>`;
}

let prevLastError: string | null = null;

/** Adds the WHAT TURNS IT INTO TEXT group to the end of `container`. Nothing is
 *  read while `windowHidden`; the returned lifecycle stops and restarts the poll. */
export function renderDictationMode(
  container: HTMLElement,
  settings: UserSettings,
  windowHidden = false,
): TabLifecycle {
  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("chip")}WHAT TURNS IT INTO TEXT</div>
    <div class="card" role="radiogroup" aria-label="What turns it into text">
      ${modeRowHtml("mode-cloud", "cloud", "Cloud")}
      ${modeRowHtml("mode-local", "chip", "Local model")}
    </div>
  `,
  );

  const cloudRow = container.querySelector<HTMLButtonElement>("#mode-cloud")!;
  const localRowEl = container.querySelector<HTMLButtonElement>("#mode-local")!;
  const keyMissing = cloudKeyMissing(getCloudKeyStatus());

  let currentMode: Mode = settings.stt_mode;
  let lastRead: LocalRead = null;
  let latestStatusToken = 0;

  function draw() {
    drawRow(cloudRow, currentMode === "cloud", {
      hint: keyMissing ? CLOUD_KEY_MISSING : CLOUD_HINT,
      alert: keyMissing,
      locked: false,
      disabled: false,
      action: "",
    });
    drawRow(localRowEl, currentMode === "local", localView(localRow(lastRead, currentMode === "local")));
  }

  /** Only Local's reads count, so a failure already shown is not shown again
   *  after a trip through Cloud. */
  function reportNewFailure(status: LocalSTTStatus) {
    if (currentMode !== "local") return;
    const reported = status.last_error ?? null;
    if (onIndicatorStateChange(prevLastError, reported)) notifyError(displayableError(reported)!);
    prevLastError = reported;
  }

  async function refreshStatus() {
    const token = ++latestStatusToken;
    let read: LocalRead;
    try {
      read = await api.sttLocalStatus();
    } catch {
      read = "failed";
    }
    if (isStaleStatusResponse(token, latestStatusToken)) return;
    lastRead = read;
    if (read !== "failed") reportNewFailure(read);
    draw();
  }

  async function switchMode(mode: Mode) {
    if (currentMode === mode) return;
    const previous = currentMode;
    currentMode = mode;
    draw();
    try {
      await api.setSttMode(mode);
    } catch (e) {
      currentMode = previous;
      draw();
      notifyError(e instanceof Error ? e.message : String(e));
      return;
    }
    await loadSettings().catch(() => {});
    void refreshStatus();
  }

  async function tryAgain() {
    if (currentMode === "local" && lastRead !== "failed") {
      try {
        await api.sttLocalPrewarm();
      } catch {}
    }
    await refreshStatus();
  }

  cloudRow.addEventListener("click", () => void switchMode("cloud"));
  localRowEl.addEventListener("click", () => {
    const row = localRow(lastRead, currentMode === "local");
    if (row.state === "failed") void tryAgain();
    else if (row.state === "not-installed" || row.state === "installed") void switchMode("local");
  });

  let pollInterval: ReturnType<typeof setInterval> | null = null;

  /** Start the 3 s status poll, or leave the running one alone — a second
   *  interval over the same handle would be unstoppable. */
  function startPolling() {
    if (pollInterval !== null) return;
    pollInterval = setInterval(() => void refreshStatus(), 3000);
  }

  /** Stop reading the local engine while the window is gone, and disown the
   *  read still in flight. The failure the row was last drawn from is kept,
   *  so a still-broken engine raises no fresh toast on every re-open. */
  function releaseResources() {
    if (pollInterval !== null) clearInterval(pollInterval);
    pollInterval = null;
    latestStatusToken += 1;
  }

  /** Restart the interval and read once in this same tick, so the row a
   *  returning user reads is one request old rather than one interval old. */
  function resumeResources() {
    startPolling();
    void refreshStatus();
  }

  draw();
  if (!windowHidden) resumeResources();

  return {
    destroy: () => {
      releaseResources();
      prevLastError = null;
    },
    releaseResources,
    resumeResources,
  } satisfies TabLifecycle;
}
