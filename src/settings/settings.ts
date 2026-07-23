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

// --- State ---

let currentTab = "general";
let settings: UserSettings | null = null;
let cloudStatus: CloudKeyStatus | null = null;
let destroyFn: (() => void) | null = null;
let settingsError: string | null = null;
// Written only by checkBackend(), and read *after* an await rather than
// captured before one: whoever handles a failure needs the freshest
// measurement, not the one that was true when their request left.
let backendReachable = true;

// --- DOM ---

const tabContent = document.getElementById("tab-content")!;
const navButtons = document.querySelectorAll<HTMLButtonElement>(".nav-btn");
const backendStatus = document.getElementById("backend-status")!;

// --- Tab routing ---

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

// `reachable` is the value the most recent health poll measured, never
// inferred from the exception's subclass: re-deriving a fact that was just
// observed is guessing, and guessing wrong is how a green badge and a "not
// responding" panel ended up in the same window.
//
// Every branch is past tense and ends with the remedy. The panel is the record
// of one attempt and is never repainted, so a present-tense claim becomes false
// the moment /health recovers — and the Settings window hides rather than
// closes (src-tauri/src/lib.rs:167-177), so this module is not re-executed when
// it is reopened and "reopen the window" would be false advice.
function settingsUnavailableMessage(error: unknown, reachable: boolean): string {
  // A 401 is itself proof the backend answered, so this outranks reachability.
  if (error instanceof ApiAuthError) {
    return `JustSay could not authenticate to its own backend, so it is refusing every request (401). Tauri bridge: ${bridgeDiagnosisText(error.diagnosis)}. Restart JustSay to try again.`;
  }
  if (!reachable) {
    return "The backend was not responding when this window loaded its settings. Make sure it is running, then restart JustSay.";
  }
  return `The backend answered, but loading settings failed: ${error instanceof Error ? error.message : String(error)}. Restart JustSay to try again.`;
}

// Built with DOM calls rather than innerHTML: the message can carry a
// backend-supplied error detail, and 'unsafe-inline' now sits in script-src
// (ADR 028), so an inline event-handler attribute in that string would run.
//
// `settings === null` has two meanings and they must not be conflated: with no
// `settingsError` recorded the first load is still in flight, and claiming the
// backend is down while its response is on the wire is its own kind of lie.
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

  // A failed settings load used to make every nav click a silent no-op, which
  // is how a broken app looked identical to an idle one (spec 042).
  if (!settings) {
    renderSettingsUnavailable(tabContent);
    return;
  }

  const renderFn = tabs[tabName];
  if (renderFn) {
    destroyFn = renderFn(tabContent, settings) || null;
  }
}

// --- Settings helpers ---

export async function loadSettings(): Promise<UserSettings> {
  // getSettings() keeps its fail-hard semantics — init() already catches a
  // rejection and shows the "Cannot load settings" panel. cloud-status is
  // explicitly non-fatal: a failed fetch there must not block Settings from
  // opening. On first load cloudStatus is still null (its initial value),
  // so leaving it untouched is equivalent to nulling it. But loadSettings()
  // is also re-called later in the session (e.g. models.ts's STT-engine
  // change) when cloudStatus may already hold a good cached value — a
  // transient refetch failure there must retain that value rather than
  // wipe it (Stage 3 review fix, spec 027 — same bug class as
  // saveSettings()'s cloud-status refetch below).
  const [loaded] = await Promise.all([
    api.getSettings(),
    api.cloudKeyStatus().then(
      (status) => { cloudStatus = status; },
      () => { /* retain previous cloudStatus */ },
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
      // Retain the previous cloudStatus on a failed refetch rather than
      // nulling it — nulling here would immediately mislabel an untouched
      // env-sourced row (e.g. Groq, when only Gemini was just saved) as
      // "No key set" for the rest of this render pass. The row for the key
      // that WAS just saved already renders from resp.settings (the "***"
      // stored state), not from this cache, so a stale cloudStatus can only
      // ever be wrong about a row nothing here changed.
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

// --- Health check ---

// Precedence offline > unauthorized > online: unreachable is the newest and
// most basic fact, and an auth verdict about a backend that has since gone
// away is stale. The badge reports the backend connection and nothing else —
// whether this window has its data is `#tab-content`'s job.
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

// The badge and nothing else: no gated request, no write to `#tab-content`, no
// concurrency state. `paintBackendStatus()` is called once, at the end, from a
// value measured in this same call — so the badge cannot paint a reachability
// verdict that a later measurement has already replaced.
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
    // Plain-browser dev open — Tauri IPC bridge is absent. Leave the
    // placeholder in place rather than blanking the footer.
  }
}

// --- Nav click handlers ---

navButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    // `!settings` bypasses the same-tab guard so that clicking the already-active
    // tab still paints the unavailable panel instead of doing nothing.
    if (tab && (tab !== currentTab || !settings)) {
      switchTab(tab);
    }
  });
});

// --- Init ---

// The settings load happens exactly once, here. Nothing reloads it on a timer:
// five consecutive attempts at automatic recovery each closed one interleaving
// and opened another, so the app now reports the failure and names the remedy
// instead of trying to repair itself (ADR 028).
async function init() {
  void initAppVersion();
  // `#tab-content` ships empty and only the load settling writes to it, so a
  // slow backend showed a blank pane rather than saying it was waiting.
  // `settingsError` is still null here, so this renders the loading copy.
  renderSettingsUnavailable(tabContent);
  await checkBackend();
  // Armed before the load starts: request() carries no timeout, so a
  // `GET /settings` that connects and never answers must not be able to stop
  // the badge from tracking /health.
  setInterval(checkBackend, 5000);
  try {
    await loadSettings();
    settingsError = null;
    // Not switchTab("general"): a nav click during the load already set
    // `currentTab`, and discarding it is a bug nobody would ever find.
    switchTab(currentTab);
  } catch (e) {
    // `backendReachable` is read here, after the await — the freshest
    // measurement is whatever the most recent poll wrote, not whatever was
    // true when this request left.
    settingsError = settingsUnavailableMessage(e, backendReachable);
    renderSettingsUnavailable(tabContent);
    // Repaint the badge now, not on the next 5 s poll: checkBackend() above ran
    // before this first gated request, so it could not yet see a 401. Without
    // this, a health-200 / settings-401 start leaves the badge green for up to
    // one poll while the panel already says "could not authenticate" — the
    // exact green-badge-over-unauthorized pair this spec exists to remove, in
    // transient form.
    paintBackendStatus(backendReachable);
    console.error("Failed to load settings:", e);
  }
}

init();
