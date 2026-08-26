import {
  acceleratorFromKeyEvent,
  detectShortcutPlatform,
  formatAccelerator,
  modifierHint,
} from "../../accelerator";
import { api, levelStream, type UserSettings } from "../../api";
import {
  EVENT_SETTINGS_CHANGED,
  EVENT_SHORTCUT_APPLIED,
  EVENT_SHORTCUT_REQUESTED,
  type ShortcutApplied,
  type ShortcutRequested,
} from "../../contracts";
import { saveSettings, getCloudKeyStatus, cachePersistedShortcut } from "../settings";
import { escapeHtml, meetingDisclosureHtml } from "../html";
import { renderKeys } from "./keys";
import { notifyError } from "../../notify";

const UPDATES_CHECK_LABEL = "Check for updates";

/** The subset of the updater plugin's `Update` this module actually uses. */
interface PendingUpdate {
  version: string;
  currentVersion: string;
  downloadAndInstall: () => Promise<unknown>;
}

/**
 * Turn an updater `check()` rejection into something a user can act on.
 *
 * The two recognised shapes both mean "the release exists but the manifest is
 * not usable yet", which reads as a broken app unless it is named.
 */
function describeUpdateCheckFailure(err: unknown): string {
  const raw = (err as Error).message ?? String(err);
  const lower = raw.toLowerCase();
  if (
    lower.includes("did not respond with a successful status code") ||
    lower.includes("could not fetch a valid release json") ||
    lower.includes("couldn't fetch a valid release json") ||
    lower.includes("couldnt fetch a valid release json")
  ) {
    return (
      "Check failed: the release manifest is not published yet. " +
      "Make sure the latest GitHub release is no longer marked as Draft."
    );
  }
  if (lower.includes("signature") || lower.includes("pubkey")) {
    return (
      "Check failed: the release manifest is not signed with the key this " +
      "build trusts. Re-run the release workflow with TAURI_SIGNING_PRIVATE_KEY set."
    );
  }
  return `Check failed: ${raw}`;
}

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
  const platform = detectShortcutPlatform(navigator);

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
        <button class="btn btn-secondary" id="btn-shortcut">${escapeHtml(formatAccelerator(settings.shortcut, platform))}</button>
      </div>
      <div class="value" id="shortcut-hint" style="padding: 4px 16px; font-size: 11px; color: var(--text-muted);">Click to change. Press new key combination, then release.</div>
    </div>

    <div class="setting-group" id="meeting-consent-group">
      ${meetingDisclosureHtml(settings.meeting_consent_acknowledged)}
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
      <div class="setting-label">History location</div>
      <div class="setting-row">
        <span class="label" style="flex: 1;">
          <input type="text" id="output-dir" value="${escapeHtml(settings.output_dir)}" style="width: 100%;" />
        </span>
        <button class="btn btn-secondary" id="btn-browse" style="margin-left: 8px;">Browse</button>
      </div>
      <div class="setting-hint">Where your transcript history is stored. If you point this at a sync folder (Dropbox / iCloud / OneDrive), large history moves may take a while.</div>
      <div id="output-dir-status" class="setting-hint" style="display: none;"></div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Recorded audio</div>
      <div class="setting-hint">Temporary voice files kept after each dictation. Your history is never stored here and is not affected.</div>
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
        <button class="btn btn-secondary" id="btn-check-updates">Check for updates</button>
      </div>
      <div class="value" id="updates-status" style="padding: 4px 16px; font-size: 11px; color: var(--text-muted);">Last checked: never.</div>
    </div>
  `;

  const langSelect = container.querySelector<HTMLSelectElement>("#lang-select")!;
  langSelect.addEventListener("change", async () => {
    try {
      await saveSettings({ language: langSelect.value });
      await emitSettingsChanged();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
    }
  });

  const keysSection = container.querySelector<HTMLElement>("#api-keys-section")!;
  const destroyKeys = renderKeys(keysSection, settings, getCloudKeyStatus());

  const btnTest = container.querySelector<HTMLButtonElement>("#btn-test-mic")!;
  const recLabel = container.querySelector<HTMLElement>("#rec-label")!;
  const levelFill = container.querySelector<HTMLElement>("#level-fill")!;

  let isRecording = false;
  let levelStreamAbort: AbortController | null = null;

  function startLevelStream(fill: HTMLElement) {
    stopLevelStream();
    levelStreamAbort = levelStream(
      (data) => {
        const pct = Math.max(0, Math.min(100, ((data.level_db + 60) / 60) * 100));
        fill.style.width = `${pct}%`;
      },
      () => {},
      () => {},
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
      } catch {}
      isRecording = false;
      btnTest.textContent = "Record";
      recLabel.textContent = "Click to test microphone";
      stopLevelStream();
      levelFill.style.width = "0%";
    } else {
      try {
        const status = await api.audioStatus();
        if (status.is_recording) {
          recLabel.textContent = "Microphone busy (widget recording)";
          return;
        }
      } catch {}

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

  const shortcutBtn = container.querySelector<HTMLButtonElement>("#btn-shortcut")!;
  const shortcutHint = container.querySelector<HTMLElement>("#shortcut-hint")!;
  let recording = false;

  const outputDir = container.querySelector<HTMLInputElement>("#output-dir")!;
  const outputStatus = container.querySelector<HTMLElement>("#output-dir-status")!;
  const btnBrowse = container.querySelector<HTMLButtonElement>("#btn-browse")!;
  const tempSize = container.querySelector<HTMLElement>("#temp-size")!;
  const btnCleanup = container.querySelector<HTMLButtonElement>("#btn-cleanup")!;

  let lastOutputDir = settings.output_dir;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let destroyed = false;
  loadFilesInfo(tempSize, () => destroyed);

  const consentGroup = container.querySelector<HTMLElement>("#meeting-consent-group")!;
  consentGroup
    .querySelector<HTMLButtonElement>("#btn-meeting-consent")!
    .addEventListener("click", async (event) => {
      const button = event.currentTarget as HTMLButtonElement;
      button.disabled = true;
      try {
        await saveSettings({ meeting_consent_acknowledged: true });
        if (destroyed) return;
        consentGroup.innerHTML = meetingDisclosureHtml(true);
      } catch (e) {
        if (destroyed) return;
        button.disabled = false;
        notifyError(e instanceof Error ? e.message : String(e));
      }
    });

  function renderStatus(text: string, kind: "warning" | "error" | "ok") {
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
        renderStatus(warning, "warning");
      } else {
        clearStatus();
      }
      loadFilesInfo(tempSize, () => destroyed);
    } catch (e) {
      if (destroyed) return;
      const msg = e instanceof Error ? e.message : String(e);
      renderStatus(msg, "error");
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

  const versionEl = container.querySelector<HTMLElement>("#app-version")!;
  const updatesBtn = container.querySelector<HTMLButtonElement>("#btn-check-updates")!;
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
  let pendingUpdate: PendingUpdate | null = null;

  function armUpdatesButton(label: string) {
    updatesBtn.textContent = label;
    updatesBtn.disabled = false;
    updatesBusy = false;
  }

  async function installPendingUpdate(update: PendingUpdate) {
    updatesBtn.textContent = "Installing…";
    updatesStatus.textContent = "Downloading and installing update…";
    try {
      await update.downloadAndInstall();
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (err) {
      updatesStatus.textContent = `Install failed: ${(err as Error).message ?? err}`;
      armUpdatesButton("Retry install");
    }
  }

  async function checkForUpdate() {
    updatesBtn.textContent = "Checking…";
    updatesStatus.textContent = "Contacting update server…";
    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const found = await check();
      if (!found) {
        updatesStatus.textContent = "You are up to date.";
        armUpdatesButton(UPDATES_CHECK_LABEL);
        return;
      }
      pendingUpdate = found;
      updatesStatus.textContent = `Update available: ${found.version} (current ${found.currentVersion}).`;
      armUpdatesButton("Install & Restart");
    } catch (err) {
      updatesStatus.textContent = describeUpdateCheckFailure(err);
      armUpdatesButton(UPDATES_CHECK_LABEL);
    }
  }

  updatesBtn.addEventListener("click", () => {
    if (updatesBusy) return;
    updatesBusy = true;
    updatesBtn.disabled = true;
    const update = pendingUpdate;
    void (update ? installPendingUpdate(update) : checkForUpdate());
  });

  async function requestShortcut(shortcut: string, revertLabelTo: string) {
    try {
      const { emit } = await loadEventApi();
      const requested: ShortcutRequested = { shortcut };
      await emit(EVENT_SHORTCUT_REQUESTED, requested);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (!destroyed) {
        shortcutHint.textContent = `Could not apply the shortcut: ${message}`;
        shortcutBtn.textContent = revertLabelTo;
      }
      notifyError(message);
    }
  }

  let captureHandler: ((e: KeyboardEvent) => void) | null = null;

  function stopCapture() {
    if (captureHandler) {
      document.removeEventListener("keydown", captureHandler, true);
      captureHandler = null;
    }
    recording = false;
  }

  shortcutBtn.addEventListener("click", () => {
    if (recording) return;
    recording = true;
    shortcutBtn.textContent = "Press keys...";
    shortcutBtn.classList.add("btn-primary");
    shortcutBtn.classList.remove("btn-secondary");
    shortcutHint.textContent = "Press your desired key combination...";

    captureHandler = (e: KeyboardEvent) => {
      if (destroyed) return;
      e.preventDefault();
      e.stopPropagation();

      const captured = acceleratorFromKeyEvent(e);
      if (!captured.ok) {
        if (captured.reason === "no-modifier") shortcutHint.textContent = modifierHint(platform);
        return;
      }

      const activeLabel = formatAccelerator(settings.shortcut, platform);

      stopCapture();
      shortcutBtn.textContent = formatAccelerator(captured.accelerator, platform);
      shortcutBtn.classList.remove("btn-primary");
      shortcutBtn.classList.add("btn-secondary");
      shortcutHint.textContent = "Applying…";

      void requestShortcut(captured.accelerator, activeLabel);
    };

    document.addEventListener("keydown", captureHandler, true);
  });

  let unlistenShortcutApplied: (() => void) | null = null;
  void (async () => {
    try {
      const { listen } = await loadEventApi();
      const unlisten = await listen<ShortcutApplied>(EVENT_SHORTCUT_APPLIED, ({ payload }) => {
        if (destroyed) return;
        const label = formatAccelerator(payload.shortcut, platform);
        if (!payload.ok) {
          shortcutHint.textContent = `${label} was not accepted: ${payload.reason ?? "unknown reason"}`;
          shortcutBtn.textContent = payload.stillActive
            ? formatAccelerator(payload.stillActive, platform)
            : "Not set";
          return;
        }
        if (payload.persisted === true) cachePersistedShortcut(payload.shortcut);
        shortcutHint.textContent =
          payload.persisted === false
            ? `${label} is active now, but could not be saved: ${payload.reason ?? "unknown reason"}`
            : `${label} is active now.`;
      });
      if (destroyed) unlisten();
      else unlistenShortcutApplied = unlisten;
    } catch {}
  })();

  return () => {
    destroyed = true;
    stopCapture();
    if (debounceTimer) clearTimeout(debounceTimer);
    stopLevelStream();
    if (isRecording) {
      api.audioStop().catch(() => {});
    }
    if (unlistenShortcutApplied) {
      unlistenShortcutApplied();
      unlistenShortcutApplied = null;
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

let eventApi: Promise<typeof import("@tauri-apps/api/event")> | null = null;

function loadEventApi(): Promise<typeof import("@tauri-apps/api/event")> {
  if (!eventApi) {
    eventApi = import("@tauri-apps/api/event").catch((e) => {
      eventApi = null;
      throw e;
    });
  }
  return eventApi;
}

async function emitSettingsChanged() {
  try {
    const { emit } = await loadEventApi();
    await emit(EVENT_SETTINGS_CHANGED);
  } catch {
  }
}
