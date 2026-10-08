import { api } from "../api";
import { ACCEPTED_AUDIO_EXTENSIONS, EVENT_FILE_PICKED, MAX_UPLOAD_BYTES } from "../contracts";
import { loadEventApi } from "../event-api";
import { icon } from "../ui/icons";

export const REFUSAL_SHOWN_MS = 4000;
export const GROW_MS = 420;

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

/** The transform that puts `box` exactly over `from`: where the drop box starts growing. */
export function transformOnto(from: DOMRect, box: DOMRect): string {
  const dx = from.left + from.width / 2 - (box.left + box.width / 2);
  const dy = from.top + from.height / 2 - (box.top + box.height / 2);
  return `translate(${dx}px, ${dy}px) scale(${from.width / box.width}, ${from.height / box.height})`;
}

function carriesFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

/**
 * Every way a file reaches the window: `pickButton` opens the system file dialog, a drag
 * carrying files dims the window under "Drop to transcribe" (ADR 087), and the shell hands
 * over a file picked from the ring or the tray. The file is sent off as a job before
 * `onStarted` runs; a refusal is explained on the overlay. Returns the teardown.
 */
export function mountFileTranscription(
  root: HTMLElement,
  pickButton: HTMLButtonElement,
  onStarted: () => void,
): () => void {
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
  const box = overlay.querySelector<HTMLElement>(".drop-overlay-box")!;
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
    if (!refused) growFromButton();
  }

  function growFromButton(): void {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const start = transformOnto(pickButton.getBoundingClientRect(), box.getBoundingClientRect());
    box.animate(
      [{ transform: start, opacity: 0 }, { opacity: 1, offset: 0.3 }, { transform: "none", opacity: 1 }],
      { duration: GROW_MS, easing: "cubic-bezier(.3, .8, .25, 1)" },
    );
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

  const onEnter = (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth += 1;
    if (depth === 1) show("Drop to transcribe", false);
  };
  const onOver = (event: DragEvent) => {
    if (carriesFiles(event)) event.preventDefault();
  };
  const onLeave = (event: DragEvent) => {
    if (!carriesFiles(event) || depth === 0) return;
    depth -= 1;
    if (depth === 0) hide();
  };
  const onDrop = (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth = 0;
    hide();
    const file = event.dataTransfer?.files?.[0];
    if (file) void send(fromPage(file));
  };

  const onPick = () => picker.click();
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

  pickButton.addEventListener("click", onPick);
  picker.addEventListener("change", onPicked);
  window.addEventListener("dragenter", onEnter);
  window.addEventListener("dragover", onOver);
  window.addEventListener("dragleave", onLeave);
  window.addEventListener("drop", onDrop);
  overlay.addEventListener("click", hide);

  return () => {
    tornDown = true;
    stopHearingShell?.();
    hide();
    window.removeEventListener("dragenter", onEnter);
    window.removeEventListener("dragover", onOver);
    window.removeEventListener("dragleave", onLeave);
    window.removeEventListener("drop", onDrop);
    pickButton.removeEventListener("click", onPick);
    overlay.remove();
    picker.remove();
  };
}
