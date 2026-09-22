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
import {
  BACKEND_WAIT_BUDGET_MS,
  nextBackendStartup,
  type BackendStartupScreen,
} from "../backend-startup";
import { EVENT_SETTINGS_HIDDEN, EVENT_SETTINGS_SHOWN } from "../contracts";
import { TimedOutError, withTimeout } from "../timeout";
import { isStaleStatusResponse } from "../stale-response";
import { nextTabAction } from "./tab-visibility";
import { renderGeneral } from "./tabs/general";
import { renderModels } from "./tabs/models";
import { renderHistory } from "./tabs/history";
import { renderMetrics } from "./tabs/metrics";
import { renderWords } from "./tabs/words";
import { renderTranscribe } from "./tabs/transcribe";


let currentTab = "general";
let settings: UserSettings | null = null;
let cloudStatus: CloudKeyStatus | null = null;
let activeTab: TabLifecycle | null = null;
let settingsError: string | null = null;
let backendReachable = true;
let settingsLoadInFlight = false;
let settingsWindowHidden = true;
let backendProbeInterval: ReturnType<typeof setInterval> | null = null;
let handledVisibilityEdges = 0;
let answeredLoadFailures = 0;
let renderedScreen: BackendStartupScreen | null = null;

const startupWaitStartedAt = Date.now();
const BACKEND_PROBE_INTERVAL_MS = 5000;


const tabContent = document.getElementById("tab-content")!;
const navButtons = document.querySelectorAll<HTMLButtonElement>(".nav-btn");
const backendStatus = document.getElementById("backend-status")!;


/** What a tab hands back so this window can let go of what it is holding.
 *
 *  `destroy` runs when the tab leaves the DOM and may tear everything down.
 *  `releaseResources` runs when the window is dismissed while the tab stays
 *  mounted, and `resumeResources` when it is shown again; each is optional, and
 *  a tab implementing one is not obliged to implement the other. */
export interface TabLifecycle {
  destroy: () => void;
  releaseResources?: () => void;
  resumeResources?: () => void;
}

type TabTeardown = (() => void) | TabLifecycle | void;

function asLifecycle(teardown: TabTeardown): TabLifecycle | null {
  if (!teardown) return null;
  return typeof teardown === "function" ? { destroy: teardown } : teardown;
}

type TabRenderer = (
  container: HTMLElement,
  settings: UserSettings,
  windowHidden: boolean,
) => TabTeardown;

const tabs: Record<string, TabRenderer> = {
  general: renderGeneral,
  models: renderModels,
  transcribe: (container) => renderTranscribe(container),
  history: (container) => renderHistory(container),
  metrics: (container) => renderMetrics(container),
  words: (container, _settings, windowHidden) => renderWords(container, windowHidden),
};

/** `bridge-missing` / `bridge-timeout` / `bridge-failed: <detail>` /
 *  `invoke-timeout` / `invoke-failed: <detail>` — the token verbatim, because
 *  these strings are what a remote user reads back to us off a screenshot and
 *  each one points at a different layer (ADR 028). All five `BridgeDiagnosis`
 *  kinds other than `ok` are covered; a list that silently omits one is worse
 *  than no list, because the omitted string then arrives off a screenshot
 *  looking like something nobody recognises. The detail is appended by asking
 *  whether the diagnosis carries one, so a sixth kind with a detail cannot be
 *  added and quietly lose it. */
function bridgeDiagnosisText(diagnosis: BridgeDiagnosis): string {
  return "detail" in diagnosis ? `${diagnosis.kind}: ${diagnosis.detail}` : diagnosis.kind;
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
    return "The backend was not responding. Make sure it is running, then try again.";
  }
  return `The backend answered, but loading settings failed: ${error instanceof Error ? error.message : String(error)}.`;
}

/** The waiting screen and the failure screen, with the way out of the second
 *  (ADR 092).
 *
 *  `starting` says the backend has not answered yet and renders the retry
 *  button disabled; `failed` names what went wrong and leaves it live. */
function renderSettingsUnavailable(container: HTMLElement, screen: BackendStartupScreen) {
  if (activeTab) {
    activeTab.destroy();
    activeTab = null;
  }
  container.innerHTML = "";
  renderedScreen = screen;

  const starting = screen === "starting";

  const title = document.createElement("div");
  title.className = "tab-title";
  title.textContent = starting ? "Starting JustSay…" : "Cannot load settings";

  const explanation = document.createElement("p");
  explanation.style.color = "var(--text-dim)";
  explanation.textContent = starting
    ? "Waiting for the backend to start."
    : settingsError ?? settingsUnavailableMessage(null, false);

  container.append(title, explanation);

  const retry = document.createElement("button");
  retry.className = "btn btn-secondary";
  retry.id = "btn-retry-settings";
  retry.textContent = "Try again";
  retry.disabled = starting || settingsLoadInFlight;
  retry.addEventListener("click", () => {
    retry.disabled = true;
    retry.textContent = "Retrying…";
    void loadSettingsIntoUi();
  });
  container.append(retry);
}

/** Load the settings into the window, or paint the screen that says why not.
 *
 *  Serialized on `settingsLoadInFlight`, so two overlapping loads cannot finish
 *  out of order and have the loser's repaint erase the winner's tab.
 *  `observedReachable` is the caller's own fresh `/health` reading; a caller
 *  without one makes this probe, because a retry must ask rather than decide on
 *  a reading up to a budget old. */
async function loadSettingsIntoUi(observedReachable: boolean | null = null): Promise<void> {
  if (settingsLoadInFlight) return;
  settingsLoadInFlight = true;
  let reachable = false;
  try {
    reachable = observedReachable ?? (await probeBackend());
    await withTimeout(loadSettings(), BACKEND_WAIT_BUDGET_MS);
    settingsError = null;
    settingsLoadInFlight = false;
    switchTab(currentTab);
    applyProbeSchedule(currentStartupScreen());
  } catch (e) {
    if (reachable) answeredLoadFailures += 1;
    settingsError = settingsUnavailableMessage(e, reachable);
    settingsLoadInFlight = false;
    const screen = currentStartupScreen();
    renderSettingsUnavailable(tabContent, screen);
    renderBackendStatus(backendReachable);
    applyProbeSchedule(screen);
    console.error("Failed to load settings:", e);
  }
}

/** The screen this window should be showing right now (ADR 092). */
function currentStartupScreen(): BackendStartupScreen {
  return nextBackendStartup({
    loaded: settings !== null,
    loadInFlight: settingsLoadInFlight,
    backendAnswering: backendReachable,
    answeredFailures: answeredLoadFailures,
    msWaiting: Date.now() - startupWaitStartedAt,
  }).screen;
}

function switchTab(tabName: string) {
  if (activeTab) {
    activeTab.destroy();
    activeTab = null;
  }

  currentTab = tabName;

  navButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  tabContent.innerHTML = "";

  if (!settings) {
    renderSettingsUnavailable(tabContent, currentStartupScreen());
    return;
  }

  const renderFn = tabs[tabName];
  if (renderFn) {
    activeTab = asLifecycle(renderFn(tabContent, settings, settingsWindowHidden));
    if (settingsWindowHidden) activeTab?.releaseResources?.();
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

let latestBackendProbeToken = 0;

/** Probes `/health`, repainting only if no newer probe has been issued since.
 *
 *  `setInterval` does not await this function, so against a backend that
 *  accepts and then goes quiet a probe outlives the 5 s interval and several
 *  overlap. Each would otherwise write `backendReachable` and repaint the badge
 *  in fetch-completion order rather than start order, so a stale probe's
 *  failure lands on top of a fresh probe's success and the header says the
 *  backend is offline while the window is reading it.
 *
 *  The guard is a generation counter and not an in-flight boolean because a
 *  boolean drops the probe instead of the answer: with a 15 s budget on a 5 s
 *  interval one is in flight essentially always, so the retry button would
 *  return without asking anything and decide on a reading up to 15 s old. Every
 *  caller here gets its own probe; only the superseded answers are discarded,
 *  so the badge reads whichever probe most recently finished rather than
 *  whichever most recently started.
 *
 *  What the guard governs is the *badge*, and the returned value is deliberately
 *  outside it. `loadSettingsIntoUi` awaits this probe to decide which failure
 *  sentence the screen shows, and discarding a superseded answer there would
 *  hand it the module-level `backendReachable` — a reading some other probe
 *  took, possibly before this window ever asked. The caller gets what its own
 *  probe observed, always; only the repaint is arbitrated. */
async function probeBackend(): Promise<boolean> {
  const token = ++latestBackendProbeToken;
  let reachable: boolean;
  try {
    await api.health();
    reachable = true;
  } catch {
    reachable = false;
  }
  if (isStaleStatusResponse(token, latestBackendProbeToken)) return reachable;
  backendReachable = reachable;
  renderBackendStatus(backendReachable);

  const decision = nextBackendStartup({
    loaded: settings !== null,
    loadInFlight: settingsLoadInFlight,
    backendAnswering: reachable,
    answeredFailures: answeredLoadFailures,
    msWaiting: Date.now() - startupWaitStartedAt,
  });
  if (
    !settingsLoadInFlight &&
    decision.screen !== "ready" &&
    decision.screen !== renderedScreen
  ) {
    renderSettingsUnavailable(tabContent, decision.screen);
  }
  if (decision.load) void loadSettingsIntoUi(reachable);
  else applyProbeSchedule(decision.screen);
  return reachable;
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


/** Cancel a drag carrying files that no element in the page handled (ADR 087).
 *
 *  An unhandled file drop navigates the webview to the dropped file. A drag
 *  carrying anything else is left to its browser default. */
function swallowUnhandledFileDrop(event: Event) {
  const dragged = (event as DragEvent).dataTransfer;
  if (!dragged || !Array.from(dragged.types).includes("Files")) return;
  event.preventDefault();
}

window.addEventListener("dragover", swallowUnhandledFileDrop);
window.addEventListener("drop", swallowUnhandledFileDrop);


/** Start the `/health` interval, or leave the running one alone. */
function startBackendProbe() {
  if (backendProbeInterval !== null) return;
  backendProbeInterval = setInterval(probeBackend, BACKEND_PROBE_INTERVAL_MS);
}

/** Stop the `/health` interval and disown whatever probe is still in flight,
 *  so no answer arriving afterwards repaints a badge nobody is looking at. */
function stopBackendProbe() {
  latestBackendProbeToken += 1;
  if (backendProbeInterval === null) return;
  clearInterval(backendProbeInterval);
  backendProbeInterval = null;
}

/** Poll `/health` while this window is on screen or still waiting for its first
 *  answer, and stop otherwise (ADR 092). */
function applyProbeSchedule(screen: BackendStartupScreen) {
  if (!settingsWindowHidden || screen === "starting") startBackendProbe();
  else stopBackendProbe();
}

/** Hand this window's periodic work, and the active tab's, to the edge that
 *  just arrived (ADR 089).
 *
 *  A resume probes once before restarting the interval, so a returning user
 *  reads a badge one request old rather than one interval old. A dismissal
 *  leaves the poll running while the window is still starting. */
function applyWindowVisibility(event: "hidden" | "shown") {
  const action = nextTabAction(event, settingsWindowHidden);
  if (action === "ignore") return;
  settingsWindowHidden = action === "release";
  if (action === "release") {
    applyProbeSchedule(currentStartupScreen());
    activeTab?.releaseResources?.();
    return;
  }
  void probeBackend();
  applyProbeSchedule(currentStartupScreen());
  activeTab?.resumeResources?.();
}

/** Route one announcement from the shell through the gate, and count it, so a
 *  visibility read in flight can tell its answer has been overtaken. */
function handleVisibilityEdge(event: "hidden" | "shown") {
  handledVisibilityEdges += 1;
  applyWindowVisibility(event);
}

/** Ask the shell what this window currently is, and route the answer through
 *  the same gate an announcement takes.
 *
 *  A show announced before the subscription attached is gone, and the window's
 *  own state (`@tauri-apps/api` 2.10) is the reading that replaces the guess.
 *  It takes both terms the shell's predicate takes, because Windows reports a
 *  minimised window as visible and reading `isVisible()` alone would resume
 *  polling on a window sitting in the taskbar. An announcement handled while
 *  the read was away is newer, so it wins. */
async function readWindowVisibility() {
  const edgesBefore = handledVisibilityEdges;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  const settingsWindow = getCurrentWindow();
  const [visible, minimized] = await Promise.all([
    settingsWindow.isVisible(),
    settingsWindow.isMinimized(),
  ]);
  if (handledVisibilityEdges !== edgesBefore) return;
  applyWindowVisibility(visible && !minimized ? "shown" : "hidden");
}

/** Follow the Settings window between dismissed and shown again.
 *
 *  The show is subscribed before the dismissal, so a partial attach keeps the
 *  edge that resumes. A page holding no show edge can never hear the window
 *  arrive, whatever the window says it is, so it polls: that is the settings
 *  screen served by Vite, and believing otherwise silences it for good. */
async function trackTabWindowVisibility() {
  let showEdgeAttached = false;
  try {
    const { listen } = await import("@tauri-apps/api/event");
    await listen(EVENT_SETTINGS_SHOWN, () => handleVisibilityEdge("shown"));
    showEdgeAttached = true;
    await listen(EVENT_SETTINGS_HIDDEN, () => handleVisibilityEdge("hidden"));
  } catch {
  }
  if (!showEdgeAttached) {
    applyWindowVisibility("shown");
    return;
  }
  try {
    await readWindowVisibility();
  } catch {
  }
}


function init() {
  void initAppVersion();
  void trackTabWindowVisibility();
  renderSettingsUnavailable(tabContent, currentStartupScreen());
  void probeBackend();
  applyProbeSchedule(currentStartupScreen());
}

init();
