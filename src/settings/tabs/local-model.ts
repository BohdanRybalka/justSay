/**
 * What the Local model row reads from the backend: the engine's status every 3 s, and the model
 * download every second while one runs. The backend holds the state, so every page that draws the
 * row shows the same one, and a download goes on while the window is closed.
 */
import { api, type LocalSTTStatus, type ModelDownload } from "../../api";
import { notifyError } from "../../notify";
import { isStaleStatusResponse } from "../../stale-response";
import { localRow, type LocalRead, type LocalRow } from "./mode-rows";

const STATUS_POLL_MS = 3000;
const DOWNLOAD_POLL_MS = 1000;

export interface LocalModelHooks {
  /** Called after every change to what the row should show. */
  onChange: () => void;
  /** Called with each status the backend answered, before `onChange`. */
  onStatus?: (status: LocalSTTStatus) => void;
}

export interface LocalModel {
  read: () => LocalRead;
  row: (selected: boolean) => LocalRow;
  refresh: () => Promise<void>;
  /** Starts the download, or resumes a failed one; the backend ignores it while one runs. */
  install: () => Promise<void>;
  resume: () => void;
  /** Stops reading and disowns an answer still in flight; a download keeps going. */
  release: () => void;
}

export function watchLocalModel(hooks: LocalModelHooks): LocalModel {
  let lastRead: LocalRead = null;
  let download: ModelDownload | null = null;
  let latestToken = 0;
  let active = false;
  let statusTimer: ReturnType<typeof setInterval> | null = null;
  let downloadTimer: ReturnType<typeof setInterval> | null = null;

  function syncDownloadTimer() {
    const wanted = active && download?.stage === "downloading";
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
    download = progress;
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
    download = next;
    if (next.stage === "downloading") hooks.onChange();
    else void refresh();
  }

  async function install() {
    try {
      download = await api.sttLocalModelStart();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
      return;
    }
    syncDownloadTimer();
    hooks.onChange();
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
    syncDownloadTimer();
    latestToken += 1;
  }

  return {
    read: () => lastRead,
    row: (selected) => localRow(lastRead, selected, download),
    refresh,
    install,
    resume,
    release,
  };
}
