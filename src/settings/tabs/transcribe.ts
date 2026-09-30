import { api } from "../../api";
import { ACCEPTED_AUDIO_EXTENSIONS, MAX_UPLOAD_BYTES } from "../../contracts";

const ACCEPT_ATTR = ACCEPTED_AUDIO_EXTENSIONS.join(",");
const BYTES_PER_MB = 1024 * 1024;
const MAX_MB = MAX_UPLOAD_BYTES / BYTES_PER_MB;

type TranscribeUiState = "idle" | "loading" | "error";

/** The drop zone above History. A picked or dropped file is sent off as a job, `onStarted`
 *  runs, and History shows it from there; only reading and sending failures show here. */
export function renderTranscribe(container: HTMLElement, onStarted: () => void): () => void {
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
          or <button type="button" class="link-btn" id="btn-pick">choose a file</button>
          — wav · mp3 · m4a · mp4 · ogg · flac · webm · aac · opus · aiff · wma (≤ ${MAX_MB} MB)
        </div>
        <input type="file" id="file-input" accept="${ACCEPT_ATTR}" hidden />
      </div>
    </div>

    <div class="setting-group" id="result-group" style="display:none;">
      <div class="transcribe-result" id="result-card">
        <div class="result-status" id="result-status"></div>
      </div>
    </div>
  `;

  const dropzone = container.querySelector<HTMLDivElement>("#dropzone")!;
  const fileInput = container.querySelector<HTMLInputElement>("#file-input")!;
  const pickBtn = container.querySelector<HTMLButtonElement>("#btn-pick")!;
  const resultGroup = container.querySelector<HTMLElement>("#result-group")!;
  const resultStatus = container.querySelector<HTMLElement>("#result-status")!;

  let busy = false;
  let destroyed = false;

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
      const enteredNext = (e as DragEvent).relatedTarget as Node | null;
      if (evt === "dragleave" && enteredNext && dropzone.contains(enteredNext)) return;
      dropzone.classList.remove("active");
    });
  });
  dropzone.addEventListener("drop", async (e) => {
    if (busy) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      await handleFile(file);
    } else {
      const dragged = e.dataTransfer?.getData("text/plain");
      if (dragged) {
        renderError("That drag carried text, not a file. Drop an audio file or use the picker.");
      }
    }
  });

  function renderUiState(state: TranscribeUiState, message = "") {
    dropzone.classList.toggle("busy", state === "loading");
    busy = state === "loading";
    resultGroup.style.display = state === "idle" ? "none" : "block";
    resultStatus.textContent = message;
    resultStatus.className = `result-status ${state === "error" ? "error" : "pending"}`;
  }

  function renderError(msg: string) {
    renderUiState("error", msg);
  }

  async function handleFile(file: File) {
    if (!validateExtension(file.name)) {
      renderError(`Unsupported format: ${file.name.split(".").pop()}`);
      return;
    }
    if (file.size === 0) {
      renderError("Empty file");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      renderError(`File too large (${(file.size / BYTES_PER_MB).toFixed(1)} MB > ${MAX_MB} MB limit)`);
      return;
    }

    renderUiState("loading", `Reading ${file.name} (${(file.size / BYTES_PER_MB).toFixed(1)} MB)...`);
    let buf: ArrayBuffer;
    try {
      buf = await file.arrayBuffer();
    } catch (e) {
      if (destroyed) return;
      renderError(`Failed to read file: ${(e as Error).message}`);
      return;
    }
    if (destroyed) return;

    try {
      await api.startFileJob(buf, file.name);
    } catch (e) {
      if (destroyed) return;
      renderError((e as Error).message);
      return;
    }
    if (destroyed) return;
    renderUiState("idle");
    onStarted();
  }

  return () => {
    destroyed = true;
  };
}

function validateExtension(filename: string): boolean {
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return false;
  const ext = filename.slice(dot).toLowerCase();
  return ACCEPTED_AUDIO_EXTENSIONS.includes(ext);
}
