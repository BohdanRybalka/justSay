/**
 * What the Local model row reads from the backend: the engine's status every 3 s, and the install
 * (download, check, engine start) every second while one runs. The backend holds the state, so
 * every page that draws the row shows the same one, and an install goes on while the window is
 * closed. An install pressed here ends with a pause for the finished steps, then `onInstalled`.
 */
import { api, type LocalSTTStatus, type ModelDownload } from "../../api";
import { notifyError } from "../../notify";
import { isStaleStatusResponse } from "../../stale-response";
import { isInstalling, localRow, type LocalRead, type LocalRow } from "./mode-rows";

const STATUS_POLL_MS = 3000;
const DOWNLOAD_POLL_MS = 1000;
const SWITCH_DELAY_MS = 900;

export interface LocalModelHooks {
  /** Called after every change to what the row should show. */
  onChange: () => void;
  /** Called with each status the backend answered, before `onChange`. */
  onStatus?: (status: LocalSTTStatus) => void;
  /** Called about a second after an install pressed here finished: the moment to choose Local. */
  onInstalled?: () => void;
}

export interface LocalModel {
  read: () => LocalRead;
  row: (selected: boolean) => LocalRow;
  refresh: () => Promise<void>;
  /** Starts the install, or resumes a failed one; the backend ignores it while one runs. */
  install: () => Promise<void>;
  /** What pressing the row does: install, retry, or hand over to `choose` once it is installed. */
  press: (selected: boolean, actions: { choose: () => void; retry: () => void }) => void;
  resume: () => void;
  /** The user chose another engine meanwhile: an install pressed here no longer ends in `onInstalled`. */
  declineSwitch: () => void;
  /** Stops reading and disowns an answer still in flight; an install keeps going. */
  release: () => void;
}

export function watchLocalModel(hooks: LocalModelHooks): LocalModel {
  let lastRead: LocalRead = null;
  let download: ModelDownload | null = null;
  let latestToken = 0;
  let active = false;
  let installingHere = false;
  let declines = 0;
  let finishing = false;
  let statusTimer: ReturnType<typeof setInterval> | null = null;
  let downloadTimer: ReturnType<typeof setInterval> | null = null;
  let switchTimer: ReturnType<typeof setTimeout> | null = null;

  function adopt(next: ModelDownload | null) {
    download = next;
    if (!installingHere) return;
    if (next?.stage === "done") finishInstall();
    else if (next?.stage === "failed") installingHere = false;
  }

  function declineSwitch() {
    declines += 1;
    installingHere = false;
    finishing = false;
    if (switchTimer !== null) clearTimeout(switchTimer);
    switchTimer = null;
  }

  function finishInstall() {
    installingHere = false;
    finishing = true;
    switchTimer = setTimeout(() => {
      switchTimer = null;
      finishing = false;
      hooks.onChange();
      hooks.onInstalled?.();
    }, SWITCH_DELAY_MS);
  }

  function syncDownloadTimer() {
    const wanted = active && download !== null && isInstalling(download.stage);
    if (wanted && downloadTimer === null) {
      downloadTimer = setInterval(() => void readDownload(), DOWNLOAD_POLL_MS);
    } else if (!wanted && downloadTimer !== null) {
      clearInterval(downloadTimer);
      downloadTimer = null;
    }
  }

  async function refresh() {
    const token = ++latestToken;
    let read: LocalRead;
    let progress = download;
    try {
      read = await api.sttLocalStatus();
      if (read.model_download_reported) {
        progress = await api.sttLocalModel().catch(() => progress);
      }
    } catch {
      read = "failed";
    }
    if (isStaleStatusResponse(token, latestToken)) return;
    lastRead = read;
    adopt(progress);
    if (read !== "failed") hooks.onStatus?.(read);
    syncDownloadTimer();
    hooks.onChange();
  }

  async function readDownload() {
    const token = latestToken;
    let next: ModelDownload;
    try {
      next = await api.sttLocalModel();
    } catch {
      return;
    }
    if (isStaleStatusResponse(token, latestToken)) return;
    adopt(next);
    if (isInstalling(next.stage)) hooks.onChange();
    else void refresh();
  }

  async function install() {
    let started: ModelDownload;
    const declinedBefore = declines;
    try {
      started = await api.sttLocalModelStart();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
      return;
    }
    latestToken += 1;
    installingHere = declines === declinedBefore;
    adopt(started);
    syncDownloadTimer();
    hooks.onChange();
  }

  function press(selected: boolean, actions: { choose: () => void; retry: () => void }) {
    const row = localRow(lastRead, selected, download, finishing);
    if (row.state === "failed") void (row.cause === "install" ? install() : actions.retry());
    else if (row.state === "not-installed" && row.tracked) void install();
    else if (row.state === "not-installed" || row.state === "installed") actions.choose();
  }

  function resume() {
    active = true;
    if (statusTimer === null) statusTimer = setInterval(() => void refresh(), STATUS_POLL_MS);
    void refresh();
  }

  function release() {
    active = false;
    if (statusTimer !== null) clearInterval(statusTimer);
    statusTimer = null;
    declineSwitch();
    syncDownloadTimer();
    latestToken += 1;
  }

  return {
    read: () => lastRead,
    row: (selected) => localRow(lastRead, selected, download, finishing),
    refresh,
    install,
    press,
    resume,
    declineSwitch,
    release,
  };
}
