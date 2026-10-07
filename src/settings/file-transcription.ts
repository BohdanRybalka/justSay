import { api } from "../api";
import { ACCEPTED_AUDIO_EXTENSIONS, MAX_UPLOAD_BYTES } from "../contracts";
import { icon } from "../ui/icons";

export const REFUSAL_SHOWN_MS = 4000;

const MAX_MB = MAX_UPLOAD_BYTES / (1024 * 1024);
export const DROP_HINT = `mp3, wav, m4a, mp4 and 7 more · up to ${MAX_MB} MB`;

/** Why this file cannot be transcribed, in the words the overlay shows, or `null` when it can. */
export function refusalOf(file: File): string | null {
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

/**
 * The two ways to transcribe a file from the window: `pickButton` opens the system file
 * dialog, and a drag carrying files anywhere on it dims it under "Drop to transcribe"
 * (ADR 087). The file is sent off as a job before `onStarted` runs; a refusal is explained
 * on the overlay. Drags carrying anything else are left to the page. Returns the teardown.
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

  async function send(file: File): Promise<void> {
    const reason = refusalOf(file);
    if (reason) {
      refuse(reason);
      return;
    }
    try {
      await api.startFileJob(await file.arrayBuffer(), file.name);
    } catch (err) {
      refuse((err as Error).message);
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
    if (file) void send(file);
  };

  const onPick = () => picker.click();
  const onPicked = () => {
    const file = picker.files?.[0];
    picker.value = "";
    if (file) void send(file);
  };

  pickButton.addEventListener("click", onPick);
  picker.addEventListener("change", onPicked);
  window.addEventListener("dragenter", onEnter);
  window.addEventListener("dragover", onOver);
  window.addEventListener("dragleave", onLeave);
  window.addEventListener("drop", onDrop);
  overlay.addEventListener("click", hide);

  return () => {
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
