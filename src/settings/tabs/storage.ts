import { api, type UserSettings } from "../../api";
import { saveSettings } from "../settings";

export function renderStorage(container: HTMLElement, settings: UserSettings): void {
  container.innerHTML = `
    <h2 class="tab-title">Storage</h2>

    <div class="setting-group">
      <div class="setting-label">Output Directory</div>
      <div class="setting-row">
        <span class="label" style="flex: 1;">
          <input type="text" id="output-dir" value="${escapeHtml(settings.output_dir)}" style="width: 100%;" />
        </span>
        <button class="btn btn-secondary" id="btn-browse" style="margin-left: 8px;">Browse</button>
      </div>
      <div class="setting-hint">If you point this at a sync folder (Dropbox / iCloud / OneDrive), large history moves may take a while.</div>
      <div id="output-dir-status" class="setting-hint" style="display: none;"></div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Transcript History</div>
      <div class="setting-row">
        <span class="label">Path</span>
        <span class="value" id="history-path-display">Loading...</span>
      </div>
      <div class="setting-row">
        <span class="label">Entries</span>
        <span class="value" id="history-entries-display">...</span>
      </div>
      <div class="setting-row" style="justify-content: flex-end;">
        <button class="btn btn-secondary" id="btn-reveal-history">Copy Path</button>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Temporary Files</div>
      <div class="setting-row">
        <span class="label">Location</span>
        <span class="value" id="temp-dir">Loading...</span>
      </div>
      <div class="setting-row">
        <div>
          <span class="label">Size</span>
          <span class="value" id="temp-size" style="margin-left: 8px;">...</span>
        </div>
        <button class="btn btn-danger" id="btn-cleanup">Clear Temp Files</button>
      </div>
    </div>
  `;

  const outputDir = container.querySelector<HTMLInputElement>("#output-dir")!;
  const outputStatus = container.querySelector<HTMLElement>("#output-dir-status")!;
  const btnBrowse = container.querySelector<HTMLButtonElement>("#btn-browse")!;
  const historyPathEl = container.querySelector<HTMLElement>("#history-path-display")!;
  const historyEntriesEl = container.querySelector<HTMLElement>("#history-entries-display")!;
  const btnRevealHistory = container.querySelector<HTMLButtonElement>("#btn-reveal-history")!;
  const tempDir = container.querySelector<HTMLElement>("#temp-dir")!;
  const tempSize = container.querySelector<HTMLElement>("#temp-size")!;
  const btnCleanup = container.querySelector<HTMLButtonElement>("#btn-cleanup")!;

  let lastOutputDir = settings.output_dir;
  loadStorageInfo({ historyPathEl, historyEntriesEl, tempDir, tempSize });

  function showStatus(text: string, kind: "warning" | "error" | "ok") {
    outputStatus.style.display = "block";
    outputStatus.textContent = text;
    outputStatus.style.color =
      kind === "error" ? "var(--red)"
      : kind === "warning" ? "var(--orange)"
      : "var(--green)";
  }

  function clearStatus() {
    outputStatus.style.display = "none";
    outputStatus.textContent = "";
  }

  async function persistOutputDir(value: string) {
    try {
      const { warning } = await saveSettings({ output_dir: value });
      lastOutputDir = value;
      if (warning) {
        showStatus(warning, "warning");
      } else {
        clearStatus();
      }
      loadStorageInfo({ historyPathEl, historyEntriesEl, tempDir, tempSize });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      showStatus(msg, "error");
      // Revert input to last known good value so the field reflects backend state.
      outputDir.value = lastOutputDir;
    }
  }

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  outputDir.addEventListener("input", () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => persistOutputDir(outputDir.value), 600);
  });

  btnBrowse.addEventListener("click", async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, title: "Select output directory" });
      if (selected) {
        outputDir.value = selected as string;
        persistOutputDir(selected as string);
      }
    } catch {
      outputDir.focus();
      outputDir.select();
    }
  });

  btnRevealHistory.addEventListener("click", async () => {
    try {
      // Use the configured output_dir (real path, not the masked ~/... display)
      // and append the canonical filename.
      const sep = outputDir.value.includes("\\") ? "\\" : "/";
      const fullPath = `${outputDir.value.replace(/[\\/]+$/, "")}${sep}history.jsonl`;
      await navigator.clipboard.writeText(fullPath);
      btnRevealHistory.textContent = "Path copied";
      setTimeout(() => (btnRevealHistory.textContent = "Copy Path"), 1500);
    } catch (e) {
      console.error(e);
    }
  });

  btnCleanup.addEventListener("click", async () => {
    btnCleanup.disabled = true;
    btnCleanup.textContent = "Cleaning...";
    try {
      const result = await api.cleanupTemp();
      tempSize.textContent = `Freed ${formatBytes(result.freed_bytes)}`;
      setTimeout(() => loadStorageInfo({ historyPathEl, historyEntriesEl, tempDir, tempSize }), 500);
    } catch (e) {
      tempSize.textContent = "Failed";
      console.error(e);
    } finally {
      btnCleanup.disabled = false;
      btnCleanup.textContent = "Clear Temp Files";
    }
  });
}

interface InfoTargets {
  historyPathEl: HTMLElement;
  historyEntriesEl: HTMLElement;
  tempDir: HTMLElement;
  tempSize: HTMLElement;
}

async function loadStorageInfo(t: InfoTargets) {
  try {
    const info = await api.getStorageInfo();
    t.historyPathEl.textContent = info.history_path;
    t.historyEntriesEl.textContent = `${info.history_entries}`;
    t.tempDir.textContent = info.temp_dir;
    t.tempSize.textContent = formatBytes(info.temp_size_bytes);
  } catch {
    t.historyPathEl.textContent = "Unknown";
    t.historyEntriesEl.textContent = "?";
    t.tempDir.textContent = "Unknown";
    t.tempSize.textContent = "Unknown";
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function escapeHtml(str: string): string {
  return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
