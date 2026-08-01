import { api, type DictateResponse, type UserSettings } from "../../api";
import { ACCEPTED_AUDIO_EXTENSIONS, MAX_UPLOAD_BYTES } from "../../contracts";

const ACCEPT_ATTR = ACCEPTED_AUDIO_EXTENSIONS.join(",");
const BYTES_PER_MB = 1024 * 1024;
const MAX_MB = MAX_UPLOAD_BYTES / BYTES_PER_MB;

type TranscribeUiState = "idle" | "loading" | "transcribing" | "done" | "error";

export function renderTranscribe(container: HTMLElement, settings: UserSettings): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Transcribe File</h2>

    <div class="setting-group">
      <div class="setting-label">Drop a file or pick one</div>
      <div class="dropzone" id="dropzone" tabindex="0" role="button"
           aria-label="Drop an audio file here or click to choose one">
        <div class="dropzone-icon" aria-hidden="true">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7" />
          </svg>
        </div>
        <div class="dropzone-title">Drop audio here</div>
        <div class="dropzone-sub">
          or <button type="button" class="link-btn" id="pick-btn">choose a file</button>
          — wav · mp3 · m4a · mp4 · ogg · flac · webm · aac · opus · aiff · wma (≤ ${MAX_MB} MB)
        </div>
        <input type="file" id="file-input" accept="${ACCEPT_ATTR}" hidden />
      </div>
    </div>

    <div class="setting-group" id="result-group" style="display:none;">
      <div class="setting-label">Result</div>
      <div class="transcribe-result" id="result-card">
        <div class="result-status" id="result-status"></div>
        <div class="result-text" id="result-text"></div>
        <div class="result-actions">
          <button class="btn btn-secondary btn-sm" id="copy-btn">Copy</button>
          <button class="btn btn-secondary btn-sm" id="reset-btn">Clear</button>
        </div>
      </div>
    </div>
  `;

  const dropzone = container.querySelector<HTMLDivElement>("#dropzone")!;
  const fileInput = container.querySelector<HTMLInputElement>("#file-input")!;
  const pickBtn = container.querySelector<HTMLButtonElement>("#pick-btn")!;
  const resultGroup = container.querySelector<HTMLElement>("#result-group")!;
  const resultStatus = container.querySelector<HTMLElement>("#result-status")!;
  const resultText = container.querySelector<HTMLElement>("#result-text")!;
  const copyBtn = container.querySelector<HTMLButtonElement>("#copy-btn")!;
  const resetBtn = container.querySelector<HTMLButtonElement>("#reset-btn")!;

  let busy = false;

  pickBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!busy) fileInput.click();
  });

  dropzone.addEventListener("click", () => {
    if (!busy) fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (busy) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!busy) dropzone.classList.add("active");
    });
  });
  ["dragleave", "dragend", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("active");
    });
  });
  dropzone.addEventListener("drop", async (e) => {
    if (busy) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      await handleFile(file);
    } else {
      const path = e.dataTransfer?.getData("text/plain");
      if (path) showError("Drag-drop received a path instead of a file. Use the picker instead.");
    }
  });

  let unlistenDrop: (() => void) | null = null;
  (async () => {
    try {
      const { getCurrentWebview } = await import("@tauri-apps/api/webview");
      const wv = getCurrentWebview();
      const off = await wv.onDragDropEvent((event) => {
        if (busy) return;
        if (event.payload.type === "drop" && event.payload.paths?.length) {
          handlePath(event.payload.paths[0]);
        }
      });
      unlistenDrop = () => { void off(); };
    } catch {
    }
  })();

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(resultText.textContent || "");
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    } catch (err) {
      console.error(err);
    }
  });
  resetBtn.addEventListener("click", () => {
    setUiState("idle");
    resultGroup.style.display = "none";
  });

  function setUiState(state: TranscribeUiState, message?: string) {
    dropzone.classList.toggle("busy", state === "loading" || state === "transcribing");
    busy = state === "loading" || state === "transcribing";

    switch (state) {
      case "loading":
        resultGroup.style.display = "block";
        resultStatus.textContent = message || "Reading file...";
        resultStatus.className = "result-status pending";
        resultText.textContent = "";
        break;
      case "transcribing":
        resultGroup.style.display = "block";
        resultStatus.textContent = message || "Transcribing...";
        resultStatus.className = "result-status pending";
        break;
      case "done":
        resultStatus.textContent = message || "Done";
        resultStatus.className = "result-status ok";
        break;
      case "error":
        resultGroup.style.display = "block";
        resultStatus.textContent = message || "Failed";
        resultStatus.className = "result-status error";
        break;
      case "idle":
        resultStatus.textContent = "";
        resultText.textContent = "";
        break;
    }
  }

  function showError(msg: string) {
    setUiState("error", msg);
  }

  async function handleFile(file: File) {
    if (!validateExtension(file.name)) {
      showError(`Unsupported format: ${file.name.split(".").pop()}`);
      return;
    }
    if (file.size === 0) {
      showError("Empty file");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      showError(`File too large (${(file.size / BYTES_PER_MB).toFixed(1)} MB > ${MAX_MB} MB limit)`);
      return;
    }

    setUiState("loading", `Reading ${file.name} (${(file.size / BYTES_PER_MB).toFixed(1)} MB)...`);
    let buf: ArrayBuffer;
    try {
      buf = await file.arrayBuffer();
    } catch (e) {
      showError(`Failed to read file: ${(e as Error).message}`);
      return;
    }

    await transcribe(buf, file.name);
  }

  async function handlePath(absolutePath: string) {
    const filename = absolutePath.split(/[\\/]/).pop() || "audio";
    if (!validateExtension(filename)) {
      showError(`Unsupported format: ${filename.split(".").pop()}`);
      return;
    }
    setUiState("loading", `Reading ${filename}...`);

    let bytes: Uint8Array;
    try {
      // @ts-ignore - optional Tauri plugin imported lazily
      const fs: any = await import(/* @vite-ignore */ "@tauri-apps/plugin-fs");
      bytes = await fs.readFile(absolutePath);
    } catch (e) {
      showError(`Cannot read file from disk: ${(e as Error).message}`);
      return;
    }
    if (bytes.byteLength > MAX_UPLOAD_BYTES) {
      showError(`File too large (${(bytes.byteLength / BYTES_PER_MB).toFixed(1)} MB > ${MAX_MB} MB limit)`);
      return;
    }
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    await transcribe(buf, filename);
  }

  async function transcribe(bytes: ArrayBuffer, filename: string) {
    setUiState("transcribing", `Transcribing ${filename}...`);
    try {
      const result: DictateResponse = await api.processFile(bytes, filename);
      const text = result.text || "";
      resultText.textContent = text || "(empty result)";
      const seconds = (result.duration_ms / 1000).toFixed(2);
      const copied = result.copied_to_clipboard ? " · copied to clipboard" : "";
      setUiState("done", `Done in ${seconds}s${copied}`);
    } catch (e) {
      showError((e as Error).message);
    }
  }

  return () => {
    if (unlistenDrop) {
      try { unlistenDrop(); } catch {}
    }
  };
}

function validateExtension(filename: string): boolean {
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return false;
  const ext = filename.slice(dot).toLowerCase();
  return ACCEPTED_AUDIO_EXTENSIONS.includes(ext);
}
