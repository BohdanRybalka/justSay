import { api } from "../api";
import { ACCEPTED_AUDIO_EXTENSIONS, EVENT_FILE_PICKED, MAX_UPLOAD_BYTES } from "../contracts";
import { loadEventApi } from "../event-api";
import { icon } from "../ui/icons";

export const REFUSAL_SHOWN_MS = 4000;
export const DROP_AREA_CLASS = "drop-area";

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
  bytes: () => Promise<ArrayBuffer>;
}

function fromPage(file: File): AudioToSend {
  return { name: file.name, size: file.size, bytes: () => file.arrayBuffer() };
}

function fromShell(picked: ShellPickedFile): AudioToSend {
  return {
    name: picked.name,
    size: picked.size,
    bytes: async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<ArrayBuffer>("take_picked_file", { token: picked.token });
    },
  };
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
  destroy: () => void;
}

/**
 * Every way a file reaches the window: `pick` opens the system file dialog, a drag carrying
 * files dims the window under "Drop to transcribe" (ADR 087) or, while a `.drop-area` is on
 * screen, lights that area instead, and the shell hands over a file picked from the ring or
 * the tray. The file is sent off as a job before `onStarted` runs; a refusal is explained on
 * the overlay.
 */
export function mountFileTranscription(root: HTMLElement, onStarted: () => void): FileTranscription {
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

  async function send(audio: AudioToSend): Promise<void> {
    const reason = refusalOf(audio);
    if (reason) {
      refuse(reason);
      return;
    }
    try {
      await api.startFileJob(await audio.bytes(), audio.name);
    } catch (err) {
      refuse(err instanceof Error ? err.message : String(err));
      return;
    }
    onStarted();
  }

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
    const file = event.dataTransfer?.files?.[0];
    if (file) void send(fromPage(file));
  };

  const onPicked = () => {
    const file = picker.files?.[0];
    picker.value = "";
    if (file) void send(fromPage(file));
  };

  let stopHearingShell: (() => void) | null = null;
  let tornDown = false;
  void hearShell();

  async function hearShell(): Promise<void> {
    try {
      const { listen } = await loadEventApi();
      const stop = await listen<ShellPickedFile>(EVENT_FILE_PICKED, ({ payload }) => void send(fromShell(payload)));
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
    destroy: () => {
      tornDown = true;
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
