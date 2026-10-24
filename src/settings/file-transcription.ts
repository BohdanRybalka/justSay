import { api, ApiAuthError, ApiRequestError, UPLOAD_STALL_MS } from "../api";
import { ACCEPTED_AUDIO_EXTENSIONS, EVENT_FILE_PICKED, MAX_UPLOAD_BYTES } from "../contracts";
import { loadEventApi } from "../event-api";
import { TimedOutError, withTimeout } from "../timeout";
import { icon } from "../ui/icons";

export const REFUSAL_SHOWN_MS = 4000;
export const DROP_AREA_CLASS = "drop-area";
export const UPLOAD_STOPPED = "The upload stopped";
export const ALREADY_UPLOADING = "This file is already uploading";

const DRAGGING_FILE_CLASS = "dragging-file";
const MAX_MB = MAX_UPLOAD_BYTES / (1024 * 1024);
export const DROP_HINT = `mp3, wav, m4a, mp4 and 7 more · up to ${MAX_MB} MB`;

/** A file picked in the system dialog the ring or the tray opened; `token` fetches its bytes once. */
export interface ShellPickedFile {
  token: string;
  name: string;
  size: number;
}

interface AudioToSend {
  name: string;
  size: number;
  body: () => Promise<Blob>;
}

function fromPage(file: File): AudioToSend {
  return { name: file.name, size: file.size, body: async () => file };
}

function fromShell(picked: ShellPickedFile): AudioToSend {
  let read: Promise<Blob> | null = null;
  const readOnce = async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    return new Blob([await invoke<ArrayBuffer>("take_picked_file", { token: picked.token })]);
  };
  return { name: picked.name, size: picked.size, body: () => (read ??= readOnce()) };
}

export type UploadState = "waiting" | "uploading" | "failed";

/** A file on its way to becoming a job. `error` says why a failed one failed, and
 *  `retryable` whether Try again can send it once more. */
export interface Upload {
  readonly name: string;
  readonly state: UploadState;
  readonly sentFraction: number;
  readonly error: string | null;
  readonly retryable: boolean;
}

/** The files this window is sending, in the order they go, for the Files page to draw. */
export interface Uploads {
  current(): readonly Upload[];
  /** Calls `listener` whenever an upload is added, moves on or goes; answers how to stop. */
  watch(listener: () => void): () => void;
  /** Stops a waiting or uploading file, or clears a failed one. */
  remove(upload: Upload): void;
  retry(upload: Upload): void;
}

interface Transfer extends Upload {
  key: string;
  state: UploadState;
  sentFraction: number;
  error: string | null;
  retryable: boolean;
  body: () => Promise<Blob>;
  stop: AbortController | null;
}

function keyOf(audio: Pick<AudioToSend, "name" | "size">): string {
  return `${audio.size}:${audio.name}`;
}

function uploadFailure(err: unknown): string {
  return err instanceof ApiRequestError || err instanceof ApiAuthError ? err.message : UPLOAD_STOPPED;
}

function readFailure(err: unknown): string {
  if (err instanceof TimedOutError) return UPLOAD_STOPPED;
  return err instanceof Error ? err.message : String(err);
}

/** What the window does as files move: `onTaken` when files join the queue, so the
 *  Files page is on screen; `onSent` once the backend has queued one as a job, resolving
 *  when its job card is drawn so the upload's card can give way to it. */
export interface UploadHooks {
  onTaken: () => void;
  onSent: () => Promise<void>;
}

/** Why this file cannot be transcribed, in the words the overlay shows, or `null` when it can. */
export function refusalOf(file: Pick<File, "name" | "size">): string | null {
  const dot = file.name.lastIndexOf(".");
  const ext = dot < 0 ? "" : file.name.slice(dot).toLowerCase();
  if (!ACCEPTED_AUDIO_EXTENSIONS.includes(ext)) return "That's not an audio file";
  if (file.size === 0) return "This file is empty";
  if (file.size > MAX_UPLOAD_BYTES) return `This file is over ${MAX_MB} MB`;
  return null;
}

function carriesFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

export interface FileTranscription {
  pick: () => void;
  uploads: Uploads;
  destroy: () => void;
}

/**
 * Every way a file reaches the window: `pick` opens the system file dialog, a drag carrying
 * files dims the window under "Drop to transcribe" (ADR 087) or, while a `.drop-area` is on
 * screen, lights that area instead, and the shell hands over a file picked from the ring or
 * the tray. Every file taken joins `uploads` and is sent one at a time; a lone refused or
 * already uploading file is explained on the overlay instead.
 */
export function mountFileTranscription(root: HTMLElement, hooks: UploadHooks): FileTranscription {
  const overlay = document.createElement("div");
  overlay.className = "drop-overlay";
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="drop-overlay-box">
      <span class="drop-overlay-icon">${icon("upload", "large")}</span>
      <b class="drop-overlay-title" aria-live="polite"></b>
      <span class="drop-overlay-hint"></span>
    </div>
  `;
  const title = overlay.querySelector<HTMLElement>(".drop-overlay-title")!;
  const hint = overlay.querySelector<HTMLElement>(".drop-overlay-hint")!;
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ACCEPTED_AUDIO_EXTENSIONS.join(",");
  picker.multiple = true;
  picker.hidden = true;
  root.append(overlay, picker);

  let depth = 0;
  let refusalTimer: number | null = null;

  function show(heading: string, refused: boolean): void {
    if (refusalTimer !== null) window.clearTimeout(refusalTimer);
    refusalTimer = null;
    title.textContent = heading;
    hint.textContent = DROP_HINT;
    overlay.classList.toggle("drop-overlay--refused", refused);
    overlay.hidden = false;
  }

  function hide(): void {
    if (refusalTimer !== null) window.clearTimeout(refusalTimer);
    refusalTimer = null;
    overlay.hidden = true;
  }

  function refuse(reason: string): void {
    show(reason, true);
    refusalTimer = window.setTimeout(hide, REFUSAL_SHOWN_MS);
  }

  let transfers: Transfer[] = [];
  const listeners = new Set<() => void>();
  let sending = false;

  function changed(): void {
    for (const listener of listeners) listener();
  }

  function fail(transfer: Transfer, error: string, retryable: boolean): void {
    transfer.state = "failed";
    transfer.error = error;
    transfer.retryable = retryable;
    transfer.stop = null;
    changed();
  }

  function forget(transfer: Transfer): void {
    transfers = transfers.filter((other) => other !== transfer);
    changed();
  }

  async function sendOne(transfer: Transfer): Promise<void> {
    const stop = new AbortController();
    transfer.stop = stop;
    transfer.state = "uploading";
    transfer.sentFraction = 0;
    changed();
    let body: Blob;
    try {
      body = await withTimeout(transfer.body(), UPLOAD_STALL_MS);
    } catch (err) {
      if (!stop.signal.aborted) fail(transfer, readFailure(err), false);
      return;
    }
    if (stop.signal.aborted) return;
    try {
      await api.startFileJob(body, transfer.name, {
        signal: stop.signal,
        onProgress: (sentFraction) => {
          transfer.sentFraction = sentFraction;
          changed();
        },
      });
    } catch (err) {
      if (!stop.signal.aborted) fail(transfer, uploadFailure(err), true);
      return;
    }
    await hooks.onSent().catch(() => {});
    forget(transfer);
  }

  async function sendWaiting(): Promise<void> {
    if (sending) return;
    sending = true;
    const nextWaiting = () => transfers.find((t) => t.state === "waiting");
    for (let next = nextWaiting(); next; next = nextWaiting()) await sendOne(next);
    sending = false;
  }

  function transferOf(audio: AudioToSend): Transfer {
    const reason = refusalOf(audio);
    return {
      key: keyOf(audio),
      name: audio.name,
      state: reason ? "failed" : "waiting",
      sentFraction: 0,
      error: reason,
      retryable: false,
      body: audio.body,
      stop: null,
    };
  }

  function take(files: AudioToSend[]): void {
    const lone = files.length === 1 ? refusalOf(files[0]) : null;
    if (lone) {
      refuse(lone);
      return;
    }
    let taken = false;
    for (const audio of files) {
      const same = transfers.find((t) => t.key === keyOf(audio));
      if (same && same.state !== "failed") continue;
      transfers = [...transfers.filter((t) => t !== same), transferOf(audio)];
      taken = true;
    }
    if (!taken) {
      refuse(ALREADY_UPLOADING);
      return;
    }
    changed();
    hooks.onTaken();
    void sendWaiting();
  }

  const uploads: Uploads = {
    current: () => transfers,
    watch(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    remove(upload) {
      const transfer = transfers.find((t) => t === upload);
      if (!transfer) return;
      transfer.stop?.abort();
      forget(transfer);
    },
    retry(upload) {
      const transfer = transfers.find((t) => t === upload);
      if (!transfer || transfer.state !== "failed" || !transfer.retryable) return;
      transfer.state = "waiting";
      transfer.error = null;
      transfer.sentFraction = 0;
      changed();
      void sendWaiting();
    },
  };

  function dragStarted(): void {
    if (root.querySelector(`.${DROP_AREA_CLASS}`)) {
      hide();
      root.classList.add(DRAGGING_FILE_CLASS);
    } else {
      show("Drop to transcribe", false);
    }
  }

  function dragEnded(): void {
    root.classList.remove(DRAGGING_FILE_CLASS);
    hide();
  }

  const onEnter = (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth += 1;
    if (depth === 1) dragStarted();
  };
  const onOver = (event: DragEvent) => {
    if (carriesFiles(event)) event.preventDefault();
  };
  const onLeave = (event: DragEvent) => {
    if (!carriesFiles(event) || depth === 0) return;
    depth -= 1;
    if (depth === 0) dragEnded();
  };
  const onDrop = (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth = 0;
    dragEnded();
    const files = Array.from(event.dataTransfer?.files ?? []);
    if (files.length > 0) take(files.map(fromPage));
  };

  const onPicked = () => {
    const files = Array.from(picker.files ?? []);
    picker.value = "";
    if (files.length > 0) take(files.map(fromPage));
  };

  let stopHearingShell: (() => void) | null = null;
  let tornDown = false;
  void hearShell();

  async function hearShell(): Promise<void> {
    try {
      const { listen } = await loadEventApi();
      const stop = await listen<ShellPickedFile>(EVENT_FILE_PICKED, ({ payload }) => take([fromShell(payload)]));
      if (tornDown) stop();
      else stopHearingShell = stop;
    } catch {
    }
  }

  picker.addEventListener("change", onPicked);
  window.addEventListener("dragenter", onEnter);
  window.addEventListener("dragover", onOver);
  window.addEventListener("dragleave", onLeave);
  window.addEventListener("drop", onDrop);
  overlay.addEventListener("click", hide);

  return {
    pick: () => picker.click(),
    uploads,
    destroy: () => {
      tornDown = true;
      for (const transfer of transfers) transfer.stop?.abort();
      transfers = [];
      listeners.clear();
      stopHearingShell?.();
      dragEnded();
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
      overlay.remove();
      picker.remove();
    },
  };
}
