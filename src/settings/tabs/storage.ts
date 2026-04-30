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
  const btnBrowse = container.querySelector<HTMLButtonElement>("#btn-browse")!;
  const tempDir = container.querySelector<HTMLElement>("#temp-dir")!;
  const tempSize = container.querySelector<HTMLElement>("#temp-size")!;
  const btnCleanup = container.querySelector<HTMLButtonElement>("#btn-cleanup")!;

  // Load storage info
  loadStorageInfo(tempDir, tempSize);

  // Output dir change with debounce
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  outputDir.addEventListener("input", () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      saveSettings({ output_dir: outputDir.value });
    }, 500);
  });

  // Browse button — use Tauri dialog if available
  btnBrowse.addEventListener("click", async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, title: "Select output directory" });
      if (selected) {
        outputDir.value = selected as string;
        saveSettings({ output_dir: selected as string });
      }
    } catch {
      // Not in Tauri or dialog not available — just focus the input
      outputDir.focus();
      outputDir.select();
    }
  });

  // Cleanup
  btnCleanup.addEventListener("click", async () => {
    btnCleanup.disabled = true;
    btnCleanup.textContent = "Cleaning...";
    try {
      const result = await api.cleanupTemp();
      tempSize.textContent = `Freed ${formatBytes(result.freed_bytes)}`;
      setTimeout(() => loadStorageInfo(tempDir, tempSize), 500);
    } catch (e) {
      tempSize.textContent = "Failed";
      console.error(e);
    } finally {
      btnCleanup.disabled = false;
      btnCleanup.textContent = "Clear Temp Files";
    }
  });
}

async function loadStorageInfo(tempDirEl: HTMLElement, tempSizeEl: HTMLElement) {
  try {
    const info = await api.getStorageInfo();
    tempDirEl.textContent = info.temp_dir;
    tempSizeEl.textContent = formatBytes(info.temp_size_bytes);
  } catch {
    tempDirEl.textContent = "Unknown";
    tempSizeEl.textContent = "Unknown";
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
