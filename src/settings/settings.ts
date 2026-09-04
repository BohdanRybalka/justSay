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
import { TimedOutError, withTimeout } from "../timeout";
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
let settingsLoadInFlight = false;

const SETTINGS_LOAD_TIMEOUT_MS = 40_000;


const tabContent = document.getElementById("tab-content")!;
const navButtons = document.querySelectorAll<HTMLButtonElement>(".nav-btn");
const backendStatus = document.getElementById("backend-status")!;


const tabs: Record<string, (container: HTMLElement, settings: UserSettings) => (() => void) | void> = {
  general: renderGeneral,
  models: renderModels,
  transcribe: (container) => renderTranscribe(container),
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

/** `TimedOutError` reaches this screen from two mechanisms and they know
 *  different things, which `subject` is what distinguishes. With a subject it
 *  came from `api.ts`, where a single request was accepted by the backend and
 *  went unanswered, and the endpoint and the budget are the facts worth
 *  reporting — the error's own sentence says them, so the prefix must not say
 *  them again. With `subject === null` it came from the outer
 *  `withTimeout(loadSettings(), ...)`, which wraps several awaits and can name
 *  none of them.
 *
 *  Neither branch claims the backend accepted anything. A budget that names an
 *  endpoint can still have expired before the request was sent — it covers the
 *  token wait too — so "accepted and never answered" would be an invention on
 *  the exact failure this text exists to explain. */
function settingsUnavailableMessage(error: unknown, reachable: boolean): string {
  if (error instanceof ApiAuthError) {
    return `JustSay could not authenticate to its own backend, so it is refusing every request (401). Tauri bridge: ${bridgeDiagnosisText(error.diagnosis)}.`;
  }
  if (error instanceof TimedOutError) {
    return error.subject === null
      ? `Loading settings did not finish in time: ${error.message}. It may still be starting up — try again.`
      : `The backend did not answer this window's request in time (${error.subject}, ${error.budgetMs / 1000} s). It may still be starting up — try again.`;
  }
  if (!reachable) {
    return "The backend was not responding when this window loaded its settings. Make sure it is running, then try again.";
  }
  return `The backend answered, but loading settings failed: ${error instanceof Error ? error.message : String(error)}.`;
}

/**
 * The failure screen, with the way out of it.
 *
 * `init()` races the sidecar, which the Rust side budgets thirty seconds for,
 * so this screen is reached on an ordinary cold start. Closing the window does
 * not reload the webview -- `src-tauri/src/lib.rs` intercepts CloseRequested
 * and hides it instead -- so without a retry the only recovery is restarting
 * the whole app.
 */
function renderSettingsUnavailable(container: HTMLElement) {
  if (destroyFn) {
    destroyFn();
    destroyFn = null;
  }
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

  const retry = document.createElement("button");
  retry.className = "btn btn-secondary";
  retry.id = "btn-retry-settings";
  retry.textContent = "Try again";
  retry.disabled = loading || settingsLoadInFlight;
  retry.addEventListener("click", () => {
    retry.disabled = true;
    retry.textContent = "Retrying…";
    void loadSettingsIntoUi();
  });
  container.append(retry);
}

/**
 * The one place settings are loaded into the window, and the only way back out
 * of the failure screen.
 *
 * Serialized on `settingsLoadInFlight` because the retry button is reachable
 * from the loading screen too: two overlapping loads could otherwise finish out
 * of order, and the loser's failure repaint would erase a tab the winner had
 * already rendered.
 *
 * The outer race is no longer what bounds an unanswered *request*. Every call
 * either half of this function makes carries its own budget, and that budget
 * now starts before the token is asked for rather than after it, so the one
 * unbounded step that used to sit in front of it — the dynamic
 * `import("@tauri-apps/api/core")` — is inside a budget too. `checkBackend()`
 * is awaited outside `loadSettings()` and therefore outside the race; it is
 * bounded because the request underneath it is, not because of anything here.
 *
 * The race is kept for what it still covers: this function grows more awaits
 * over time, and it is the only thing that bounds a step nobody remembered to
 * give a budget of its own. Its 40 s sits above the ~18 s worst case of the two
 * parallel bounded calls, so it fires only on something new.
 */
async function loadSettingsIntoUi(): Promise<void> {
  if (settingsLoadInFlight) return;
  settingsLoadInFlight = true;
  try {
    await checkBackend();
    await withTimeout(loadSettings(), SETTINGS_LOAD_TIMEOUT_MS);
    settingsError = null;
    settingsLoadInFlight = false;
    switchTab(currentTab);
  } catch (e) {
    settingsError = settingsUnavailableMessage(e, backendReachable);
    settingsLoadInFlight = false;
    renderSettingsUnavailable(tabContent);
    renderBackendStatus(backendReachable);
    console.error("Failed to load settings:", e);
  }
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

export function cachePersistedShortcut(shortcut: string): void {
  if (settings) settings = { ...settings, shortcut };
}

export function getCloudKeyStatus(): CloudKeyStatus | null {
  return cloudStatus;
}


function renderBackendStatus(reachable: boolean) {
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
  renderBackendStatus(backendReachable);
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
  setInterval(checkBackend, 5000);
  await loadSettingsIntoUi();
}

init();
