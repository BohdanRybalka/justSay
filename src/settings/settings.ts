import { api, type UserSettings } from "../api";
import { renderGeneral } from "./tabs/general";
import { renderModels } from "./tabs/models";
import { renderAudio } from "./tabs/audio";
import { renderStorage } from "./tabs/storage";
import { renderHistory } from "./tabs/history";
import { renderMetrics } from "./tabs/metrics";

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
  storage: renderStorage,
  history: (container) => renderHistory(container),
  metrics: (container) => renderMetrics(container),
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

export async function saveSettings(updates: Partial<UserSettings>): Promise<UserSettings> {
  settings = await api.updateSettings(updates);
  return settings;
}

export function getSettings(): UserSettings | null {
  return settings;
}

// --- Health check ---

async function checkBackend() {
  try {
    const health = await api.health();
    backendStatus.textContent = `Backend: ${health.version}`;
    backendStatus.className = "status-indicator online";
  } catch {
    backendStatus.textContent = "Backend: Offline";
    backendStatus.className = "status-indicator offline";
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
