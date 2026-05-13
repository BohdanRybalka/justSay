import { getVersion } from "@tauri-apps/api/app";
import { api, type UserSettings } from "../api";
import { renderGeneral } from "./tabs/general";
import { renderModels } from "./tabs/models";
import { renderAudio } from "./tabs/audio";
import { renderStorage } from "./tabs/storage";
import { renderHistory } from "./tabs/history";
import { renderMetrics } from "./tabs/metrics";
import { renderWords } from "./tabs/words";
import { renderTranscribe } from "./tabs/transcribe";
import { renderKeys } from "./tabs/keys";

// --- State ---

let currentTab = "general";
let settings: UserSettings | null = null;
let destroyFn: (() => void) | null = null;

// --- DOM ---

const tabContent = document.getElementById("tab-content")!;
const navButtons = document.querySelectorAll<HTMLButtonElement>(".nav-btn");
const backendStatus = document.getElementById("backend-status")!;

// --- Tab routing ---

const tabs: Record<string, (container: HTMLElement, settings: UserSettings) => (() => void) | void> = {
  general: renderGeneral,
  models: renderModels,
  audio: renderAudio,
  transcribe: renderTranscribe,
  keys: renderKeys,
  storage: renderStorage,
  history: (container) => renderHistory(container),
  metrics: (container) => renderMetrics(container),
  words: (container) => renderWords(container),
};

function switchTab(tabName: string) {
  if (!settings) return;

  // Cleanup previous tab
  if (destroyFn) {
    destroyFn();
    destroyFn = null;
  }

  currentTab = tabName;

  // Update nav
  navButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  // Render new tab
  tabContent.innerHTML = "";
  const renderFn = tabs[tabName];
  if (renderFn) {
    destroyFn = renderFn(tabContent, settings) || null;
  }
}

// --- Settings helpers ---

export async function loadSettings(): Promise<UserSettings> {
  settings = await api.getSettings();
  return settings;
}

export async function saveSettings(updates: Partial<UserSettings>): Promise<{ settings: UserSettings; warning: string | null }> {
  const resp = await api.updateSettings(updates);
  settings = resp.settings;
  return { settings: resp.settings, warning: resp.warning };
}

export function getSettings(): UserSettings | null {
  return settings;
}

// --- Health check ---

async function checkBackend() {
  try {
    await api.health();
    backendStatus.textContent = "Backend";
    backendStatus.className = "status-indicator online";
  } catch {
    backendStatus.textContent = "Backend offline";
    backendStatus.className = "status-indicator offline";
  }
}

async function initAppVersion() {
  const el = document.getElementById("sidebar-version");
  if (!el) return;
  try {
    const v = await getVersion();
    el.textContent = `v${v}`;
  } catch {
    // Plain-browser dev open — Tauri IPC bridge is absent. Leave the
    // placeholder in place rather than blanking the footer.
  }
}

// --- Nav click handlers ---

navButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    if (tab && tab !== currentTab) {
      switchTab(tab);
    }
  });
});

// --- Init ---

async function init() {
  void initAppVersion();
  await checkBackend();
  setInterval(checkBackend, 5000);

  try {
    await loadSettings();
    switchTab("general");
  } catch (e) {
    tabContent.innerHTML = `<div class="tab-title">Cannot load settings</div><p style="color: var(--text-dim)">Backend is not responding. Make sure it is running.</p>`;
    console.error("Failed to load settings:", e);
  }
}

init();
