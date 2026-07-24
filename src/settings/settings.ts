import { getVersion } from "@tauri-apps/api/app";
import {
  api,
  ApiAuthError,
  lastBridgeDiagnosis,
  sawAuthFailure,
  type BridgeDiagnosis,
  type CloudKeyStatus,
  type UserSettings,
} from "../api";
import { renderGeneral } from "./tabs/general";
import { renderModels } from "./tabs/models";
import { renderHistory } from "./tabs/history";
import { renderMetrics } from "./tabs/metrics";
import { renderWords } from "./tabs/words";
import { renderTranscribe } from "./tabs/transcribe";


let currentTab = "general";
let settings: UserSettings | null = null;
let cloudStatus: CloudKeyStatus | null = null;
let destroyFn: (() => void) | null = null;
let settingsError: string | null = null;
let backendReachable = true;


const tabContent = document.getElementById("tab-content")!;
const navButtons = document.querySelectorAll<HTMLButtonElement>(".nav-btn");
const backendStatus = document.getElementById("backend-status")!;


const tabs: Record<string, (container: HTMLElement, settings: UserSettings) => (() => void) | void> = {
  general: renderGeneral,
  models: renderModels,
  transcribe: renderTranscribe,
  history: (container) => renderHistory(container),
  metrics: (container) => renderMetrics(container),
  words: (container) => renderWords(container),
};

/** `bridge-missing` / `invoke-timeout` / `invoke-failed: <detail>` — the token
 *  verbatim, because these strings are what a remote user reads back to us off
 *  a screenshot and each one points at a different layer (ADR 028). */
function bridgeDiagnosisText(diagnosis: BridgeDiagnosis): string {
  return diagnosis.kind === "invoke-failed"
    ? `invoke-failed: ${diagnosis.detail}`
    : diagnosis.kind;
}

function settingsUnavailableMessage(error: unknown, reachable: boolean): string {
  if (error instanceof ApiAuthError) {
    return `JustSay could not authenticate to its own backend, so it is refusing every request (401). Tauri bridge: ${bridgeDiagnosisText(error.diagnosis)}. Restart JustSay to try again.`;
  }
  if (!reachable) {
    return "The backend was not responding when this window loaded its settings. Make sure it is running, then restart JustSay.";
  }
  return `The backend answered, but loading settings failed: ${error instanceof Error ? error.message : String(error)}. Restart JustSay to try again.`;
}

function renderSettingsUnavailable(container: HTMLElement) {
  container.innerHTML = "";

  const loading = settingsError === null;

  const title = document.createElement("div");
  title.className = "tab-title";
  title.textContent = loading ? "Loading settings..." : "Cannot load settings";

  const explanation = document.createElement("p");
  explanation.style.color = "var(--text-dim)";
  explanation.textContent = loading
    ? "Waiting for the backend to answer."
    : settingsError;

  container.append(title, explanation);
}

function switchTab(tabName: string) {
  if (destroyFn) {
    destroyFn();
    destroyFn = null;
  }

  currentTab = tabName;

  navButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  tabContent.innerHTML = "";

  if (!settings) {
    renderSettingsUnavailable(tabContent);
    return;
  }

  const renderFn = tabs[tabName];
  if (renderFn) {
    destroyFn = renderFn(tabContent, settings) || null;
  }
}


export async function loadSettings(): Promise<UserSettings> {
  const [loaded] = await Promise.all([
    api.getSettings(),
    api.cloudKeyStatus().then(
      (status) => { cloudStatus = status; },
      () => {},
    ),
  ]);
  settings = loaded;
  return settings;
}

const KEY_FIELDS = new Set(["gemini_api_key", "groq_api_key"]);

export async function saveSettings(updates: Partial<UserSettings>): Promise<{ settings: UserSettings; warning: string | null }> {
  const resp = await api.updateSettings(updates);
  settings = resp.settings;
  if (Object.keys(updates).some((k) => KEY_FIELDS.has(k))) {
    try {
      cloudStatus = await api.cloudKeyStatus();
    } catch {
    }
  }
  return { settings: resp.settings, warning: resp.warning };
}

export function getSettings(): UserSettings | null {
  return settings;
}

export function getCloudKeyStatus(): CloudKeyStatus | null {
  return cloudStatus;
}


function paintBackendStatus(reachable: boolean) {
  if (!reachable) {
    backendStatus.textContent = "Backend offline";
    backendStatus.className = "status-indicator offline";
    backendStatus.removeAttribute("title");
    return;
  }

  if (sawAuthFailure()) {
    backendStatus.textContent = "Backend unauthorized";
    backendStatus.className = "status-indicator error";
    backendStatus.title = `Backend rejected an authenticated request (401). Tauri bridge: ${bridgeDiagnosisText(lastBridgeDiagnosis())}`;
    return;
  }

  backendStatus.textContent = "Backend";
  backendStatus.className = "status-indicator online";
  backendStatus.removeAttribute("title");
}

async function checkBackend() {
  try {
    await api.health();
    backendReachable = true;
  } catch {
    backendReachable = false;
  }
  paintBackendStatus(backendReachable);
}

async function initAppVersion() {
  const el = document.getElementById("sidebar-version");
  if (!el) return;
  try {
    const v = await getVersion();
    el.textContent = `v${v}`;
  } catch {
  }
}


navButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    if (tab && (tab !== currentTab || !settings)) {
      switchTab(tab);
    }
  });
});


async function init() {
  void initAppVersion();
  renderSettingsUnavailable(tabContent);
  await checkBackend();
  setInterval(checkBackend, 5000);
  try {
    await loadSettings();
    settingsError = null;
    switchTab(currentTab);
  } catch (e) {
    settingsError = settingsUnavailableMessage(e, backendReachable);
    renderSettingsUnavailable(tabContent);
    paintBackendStatus(backendReachable);
    console.error("Failed to load settings:", e);
  }
}

init();
