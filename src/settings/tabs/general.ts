import { api, levelStream, type UserSettings } from "../../api";
import { saveSettings, getCloudKeyStatus } from "../settings";
import { escapeHtml } from "../html";
import { renderKeys } from "./keys";
import { notifyError } from "../../notify";

const LANGUAGES = [
  { code: "uk", label: "Ukrainian" },
  { code: "en", label: "English" },
  { code: "de", label: "German" },
  { code: "fr", label: "French" },
  { code: "es", label: "Spanish" },
  { code: "pl", label: "Polish" },
  { code: "ja", label: "Japanese" },
  { code: "zh", label: "Chinese" },
];

export function renderGeneral(container: HTMLElement, settings: UserSettings): () => void {
  container.innerHTML = `
    <h2 class="tab-title">General</h2>

    <div class="setting-group">
      <div class="setting-label">Dictation Language</div>
      <div class="setting-row">
        <span class="label">Language</span>
        <select id="lang-select">
          ${LANGUAGES.map(
            (l) => `<option value="${l.code}" ${l.code === settings.language ? "selected" : ""}>${l.label}</option>`
          ).join("")}
        </select>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Global Shortcut</div>
      <div class="setting-row">
        <span class="label">Push-to-talk</span>
        <button class="btn btn-secondary" id="shortcut-btn">${formatShortcut(settings.shortcut)}</button>
      </div>
      <div class="value" id="shortcut-hint" style="padding: 4px 16px; font-size: 11px; color: var(--text-muted);">Click to change. Press new key combination, then release.</div>
    </div>

    <div id="api-keys-section"></div>

    <div class="setting-group">
      <div class="setting-label">Microphone Test</div>
      <div class="setting-row" style="flex-direction: column; align-items: stretch; gap: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <span class="label" id="rec-label">Click to test microphone</span>
          <button class="btn btn-primary" id="btn-test-mic">Record</button>
        </div>
        <div class="level-meter">
          <div class="level-meter-fill" id="level-fill"></div>
        </div>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Files</div>
      <div class="setting-row">
        <span class="label" style="flex: 1;">
          <input type="text" id="output-dir" value="${escapeHtml(settings.output_dir)}" style="width: 100%;" />
        </span>
        <button class="btn btn-secondary" id="btn-browse" style="margin-left: 8px;">Browse</button>
      </div>
      <div class="setting-hint">If you point this at a sync folder (Dropbox / iCloud / OneDrive), large history moves may take a while.</div>
      <div id="output-dir-status" class="setting-hint" style="display: none;"></div>
      <div class="setting-row">
        <div>
          <span class="label">Size</span>
          <span class="value" id="temp-size" style="margin-left: 8px;">...</span>
        </div>
        <button class="btn btn-danger" id="btn-cleanup">Clear Temp Files</button>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">About</div>
      <div class="setting-row">
        <span class="label">Version</span>
        <span class="value" id="app-version">…</span>
      </div>
      <div class="setting-row">
        <span class="label">Updates</span>
        <button class="btn btn-secondary" id="check-updates-btn">Check for updates</button>
      </div>
      <div class="value" id="updates-status" style="padding: 4px 16px; font-size: 11px; color: var(--text-muted);">Last checked: never.</div>
    </div>
  `;

  // Language
  const langSelect = container.querySelector<HTMLSelectElement>("#lang-select")!;
  langSelect.addEventListener("change", async () => {
    try {
      await saveSettings({ language: langSelect.value });
      // The recording widget runs in a separate webview and caches the
      // language in module state — without this emit it keeps transcribing
      // in the old language until the app restarts.
      await emitSettingsChanged();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
    }
  });

  // API Keys (ported from keys.ts — rendered as an embedded subsection)
  const keysSection = container.querySelector<HTMLElement>("#api-keys-section")!;
  const destroyKeys = renderKeys(keysSection, settings, getCloudKeyStatus());

  // Microphone test
  const btnTest = container.querySelector<HTMLButtonElement>("#btn-test-mic")!;
  const recLabel = container.querySelector<HTMLElement>("#rec-label")!;
  const levelFill = container.querySelector<HTMLElement>("#level-fill")!;

  let isRecording = false;
  let levelStreamAbort: AbortController | null = null;

  function startLevelStream(fill: HTMLElement) {
    stopLevelStream();
    levelStreamAbort = levelStream(
      (data) => {
        // dBFS range: -60 (silent) to 0 (max). Map to 0-100%.
        const pct = Math.max(0, Math.min(100, ((data.level_db + 60) / 60) * 100));
        fill.style.width = `${pct}%`;
      },
      () => { /* server closed the stream because recording stopped; the
                 "Stop" click branch already resets the fill explicitly */ },
      () => { /* ignore transport errors, mirrors the old catch-all */ },
    );
  }

  function stopLevelStream() {
    if (levelStreamAbort) {
      levelStreamAbort.abort();
      levelStreamAbort = null;
    }
  }

  btnTest.addEventListener("click", async () => {
    if (isRecording) {
      try {
        await api.audioStop();
      } catch { /* ignore */ }
      isRecording = false;
      btnTest.textContent = "Record";
      recLabel.textContent = "Click to test microphone";
      stopLevelStream();
      levelFill.style.width = "0%";
    } else {
      // Check if already recording (widget might be using it)
      try {
        const status = await api.audioStatus();
        if (status.is_recording) {
          recLabel.textContent = "Microphone busy (widget recording)";
          return;
        }
      } catch { /* ignore */ }

      try {
        await api.audioStart();
        isRecording = true;
        btnTest.textContent = "Stop";
        recLabel.textContent = "Recording...";
        startLevelStream(levelFill);
      } catch (e) {
        recLabel.textContent = "Failed to start";
        console.error(e);
      }
    }
  });

  // Shortcut recorder
  const shortcutBtn = container.querySelector<HTMLButtonElement>("#shortcut-btn")!;
  const shortcutHint = container.querySelector<HTMLElement>("#shortcut-hint")!;
  let recording = false;

  // Files: output directory + temp file cleanup
  const outputDir = container.querySelector<HTMLInputElement>("#output-dir")!;
  const outputStatus = container.querySelector<HTMLElement>("#output-dir-status")!;
  const btnBrowse = container.querySelector<HTMLButtonElement>("#btn-browse")!;
  const tempSize = container.querySelector<HTMLElement>("#temp-size")!;
  const btnCleanup = container.querySelector<HTMLButtonElement>("#btn-cleanup")!;

  let lastOutputDir = settings.output_dir;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;
  loadFilesInfo(tempSize, () => destroyed);

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
      if (destroyed) return;
      lastOutputDir = value;
      if (warning) {
        showStatus(warning, "warning");
      } else {
        clearStatus();
      }
      loadFilesInfo(tempSize, () => destroyed);
    } catch (e) {
      if (destroyed) return;
      const msg = e instanceof Error ? e.message : String(e);
      showStatus(msg, "error");
      // Revert input to last known good value so the field reflects backend state.
      outputDir.value = lastOutputDir;
    }
  }

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

  btnCleanup.addEventListener("click", async () => {
    btnCleanup.disabled = true;
    btnCleanup.textContent = "Cleaning...";
    try {
      const result = await api.cleanupTemp();
      if (destroyed) return;
      tempSize.textContent = `Freed ${formatBytes(result.freed_bytes)}`;
      setTimeout(() => {
        if (!destroyed) loadFilesInfo(tempSize, () => destroyed);
      }, 500);
    } catch (e) {
      if (destroyed) return;
      tempSize.textContent = "Failed";
      console.error(e);
    } finally {
      if (!destroyed) {
        btnCleanup.disabled = false;
        btnCleanup.textContent = "Clear Temp Files";
      }
    }
  });

  // About / Updates
  const versionEl = container.querySelector<HTMLElement>("#app-version")!;
  const updatesBtn = container.querySelector<HTMLButtonElement>("#check-updates-btn")!;
  const updatesStatus = container.querySelector<HTMLElement>("#updates-status")!;

  void (async () => {
    try {
      const { getVersion } = await import("@tauri-apps/api/app");
      versionEl.textContent = await getVersion();
    } catch {
      versionEl.textContent = "unknown";
    }
  })();

  let updatesBusy = false;
  updatesBtn.addEventListener("click", async () => {
    if (updatesBusy) return;
    updatesBusy = true;
    updatesBtn.disabled = true;
    updatesBtn.textContent = "Checking…";
    updatesStatus.textContent = "Contacting update server…";

    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (!update) {
        updatesStatus.textContent = "You are up to date.";
        updatesBtn.textContent = "Check for updates";
        return;
      }
      updatesStatus.textContent = `Update available: ${update.version} (current ${update.currentVersion}).`;
      updatesBtn.textContent = "Install & Restart";
      updatesBtn.disabled = false;
      updatesBtn.onclick = async () => {
        updatesBtn.disabled = true;
        updatesBtn.textContent = "Installing…";
        updatesStatus.textContent = "Downloading and installing update…";
        try {
          await update.downloadAndInstall();
          const { relaunch } = await import("@tauri-apps/plugin-process");
          await relaunch();
        } catch (err) {
          updatesStatus.textContent = `Install failed: ${(err as Error).message ?? err}`;
          updatesBtn.textContent = "Retry install";
          updatesBtn.disabled = false;
          // Release the busy flag so a subsequent click on the same
          // (renamed) button doesn't get swallowed by the outer guard.
          updatesBusy = false;
        }
      };
    } catch (err) {
      const raw = (err as Error).message ?? String(err);
      const lower = raw.toLowerCase();
      // The Tauri updater reports two common operational failures with
      // cryptic strings that scared the user. Translate them to actionable
      // hints. Anything else falls through to the raw text so we don't
      // mask unexpected bugs.
      let friendly = `Check failed: ${raw}`;
      if (
        lower.includes("did not respond with a successful status code") ||
        lower.includes("could not fetch a valid release json") ||
        lower.includes("couldn't fetch a valid release json") ||
        lower.includes("couldnt fetch a valid release json")
      ) {
        friendly =
          "Check failed: the release manifest is not published yet. " +
          "Make sure the latest GitHub release is no longer marked as Draft.";
      } else if (lower.includes("signature") || lower.includes("pubkey")) {
        friendly =
          "Check failed: the release manifest is not signed with the key this " +
          "build trusts. Re-run the release workflow with TAURI_SIGNING_PRIVATE_KEY set.";
      }
      updatesStatus.textContent = friendly;
      updatesBtn.textContent = "Check for updates";
    } finally {
      updatesBusy = false;
      if (updatesBtn.textContent === "Checking…") {
        updatesBtn.disabled = false;
        updatesBtn.textContent = "Check for updates";
      }
    }
  });

  shortcutBtn.addEventListener("click", () => {
    if (recording) return;
    recording = true;
    shortcutBtn.textContent = "Press keys...";
    shortcutBtn.classList.add("btn-primary");
    shortcutBtn.classList.remove("btn-secondary");
    shortcutHint.textContent = "Press your desired key combination...";

    const handler = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      // Wait for a non-modifier key
      if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;

      const parts: string[] = [];
      if (e.ctrlKey) parts.push("Ctrl");
      if (e.altKey) parts.push("Alt");
      if (e.shiftKey) parts.push("Shift");
      if (e.metaKey) parts.push("Super");

      // Need at least one modifier
      if (parts.length === 0) {
        shortcutHint.textContent = "Must include at least one modifier (Ctrl, Alt, Shift)";
        return;
      }

      parts.push(e.code);
      const shortcut = parts.join("+");

      recording = false;
      shortcutBtn.textContent = formatShortcut(shortcut);
      shortcutBtn.classList.remove("btn-primary");
      shortcutBtn.classList.add("btn-secondary");
      shortcutHint.textContent = "Shortcut saved. Restart app to apply.";

      document.removeEventListener("keydown", handler, true);
      saveSettings({ shortcut });
      emitSettingsChanged();
    };

    document.addEventListener("keydown", handler, true);
  });

  return () => {
    recording = false;
    destroyed = true;
    if (debounceTimer) clearTimeout(debounceTimer);
    stopLevelStream();
    if (isRecording) {
      api.audioStop().catch(() => {});
    }
    destroyKeys();
  };
}

async function loadFilesInfo(tempSize: HTMLElement, isDestroyed: () => boolean) {
  try {
    const info = await api.getStorageInfo();
    if (isDestroyed()) return;
    tempSize.textContent = formatBytes(info.temp_size_bytes);
  } catch {
    if (!isDestroyed()) tempSize.textContent = "Unknown";
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function formatShortcut(shortcut: string): string {
  return shortcut
    .replace("Key", "")
    .replace("Digit", "")
    .replace(/\+/g, " + ");
}

async function emitSettingsChanged() {
  try {
    const { emit } = await import("@tauri-apps/api/event");
    await emit("settings-changed");
  } catch {
    // Not in Tauri
  }
}
