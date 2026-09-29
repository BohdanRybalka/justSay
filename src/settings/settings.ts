import { getVersion } from "@tauri-apps/api/app";
import {
  api,
  ApiAuthError,
  ApiRequestError,
  lastBridgeDiagnosis,
  sawAuthFailure,
  type BridgeDiagnosis,
  type CloudKeyStatus,
  type UserSettings,
} from "../api";
import {
  BACKEND_WAIT_BUDGET_MS,
  nextBackendStartup,
  type BackendStartupDecision,
  type BackendStartupScreen,
} from "../backend-startup";
import {
  EVENT_NAVIGATE_PANEL,
  EVENT_SETTINGS_CHANGED,
  EVENT_SETTINGS_HIDDEN,
  EVENT_SETTINGS_SHOWN,
  type NavigatePanel,
} from "../contracts";
import { loadEventApi } from "../event-api";
import { TimedOutError, withTimeout } from "../timeout";
import { isStaleStatusResponse } from "../stale-response";
import { nextTabAction } from "./tab-visibility";
import { renderDictation } from "./tabs/dictation";
import { renderDictationMode } from "./tabs/dictation-mode";
import { renderDictationMeetings, type MeetingsGroup } from "./tabs/dictation-meetings";
import { renderDictionary } from "./tabs/dictionary";
import { renderHistory, renderHistoryHeading } from "./tabs/history";
import { renderWords } from "./tabs/words";
import { renderTranscribe } from "./tabs/transcribe";
import { renderAccount } from "./tabs/account";
import { renderSettingsPanel } from "./tabs/settings-panel";
import { applyAppTheme, applyThemePreference } from "../ui/theme";
import { mountIconSprite } from "../ui/icons";
import { detectShortcutPlatform } from "../accelerator";
import { renderTitlebar, wireTitlebar } from "./shell/titlebar";
import { displayName, readOsDisplayName } from "./shell/account-name";
import {
  backendStateOf,
  renderAccountRow,
  renderPanelSelection,
  renderSidebarStatus,
  type PanelName,
} from "./shell/sidebar";


let currentPanel: PanelName = "insights";
let osAccountName = "";
let settings: UserSettings | null = null;
let cloudStatus: CloudKeyStatus | null = null;
let activeTab: TabLifecycle | null = null;
let settingsError: string | null = null;
let backendReachable = false;
let settingsLoadInFlight = false;
let settingsWindowHidden = true;
let backendProbeInterval: ReturnType<typeof setInterval> | null = null;
let handledVisibilityEdges = 0;
let answeredLoadFailures = 0;
let renderedUnavailable: { screen: BackendStartupScreen; retryDisabled: boolean } | null = null;

const startupWaitStartedAt = performance.now();
const BACKEND_PROBE_INTERVAL_MS = 5000;
const BACKEND_STILL_STARTING_MESSAGE = `The backend has not finished starting in ${BACKEND_WAIT_BUDGET_MS / 1000} seconds. It may still be coming up — try again, or restart JustSay.`;


const titlebar = document.getElementById("titlebar")!;
const pane = document.getElementById("pane")!;
const sidebar = document.getElementById("sidebar")!;
const sidebarStatus = document.getElementById("sidebar-status")!;


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

type PanelRenderer = (
  container: HTMLElement,
  settings: UserSettings,
  windowHidden: boolean,
) => TabTeardown;

/** Several parts of one panel answering as one. */
function combineLifecycles(parts: TabLifecycle[]): TabLifecycle {
  return {
    destroy: () => parts.forEach((part) => part.destroy()),
    releaseResources: () => parts.forEach((part) => part.releaseResources?.()),
    resumeResources: () => parts.forEach((part) => part.resumeResources?.()),
  };
}

/** Mount old tabs, whole, into a panel whose redesign has not landed yet.
 *  Each sits in its own `.legacy-tab`, the only place the old stylesheet
 *  reaches, and the panel answers for all of them as one. */
function hostLegacyTabs(
  container: HTMLElement,
  mounts: ((tab: HTMLElement) => TabTeardown)[],
): TabLifecycle {
  return combineLifecycles(
    mounts.flatMap((mount) => {
      const tab = document.createElement("div");
      tab.className = "legacy-tab";
      container.append(tab);
      return asLifecycle(mount(tab)) ?? [];
    }),
  );
}

const panels: Record<PanelName, PanelRenderer> = {
  insights: (container, _settings, windowHidden) =>
    hostLegacyTabs(container, [(tab) => renderWords(tab, windowHidden)]),
  history: (container, loaded) => {
    renderHistoryHeading(container);
    const transcribe = hostLegacyTabs(container, [renderTranscribe]);
    return combineLifecycles([transcribe, renderHistory(container, loaded)]);
  },
  dictation: (container, loaded, windowHidden) => {
    const everyday = renderDictation(container, loaded);
    let meetings: MeetingsGroup | null = null;
    const mode = renderDictationMode(container, loaded, windowHidden, (chosen) => meetings?.showMode(chosen));
    meetings = renderDictationMeetings(container, loaded);
    const dictionary = renderDictionary(container, loaded);
    return combineLifecycles([everyday, mode, meetings, dictionary]);
  },
  settings: renderSettingsPanel,
  account: (container, loaded) =>
    renderAccount(container, { chosen: loaded.display_name, osName: osAccountName }, renameUser),
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

/** The waiting screen and the failure screen, with the way out of both
 *  (ADR 092).
 *
 *  `starting` says the backend has not answered yet; `failed` names what went
 *  wrong. **Try again** is live unless a load is already in flight, so a press
 *  always issues a request. */
function renderSettingsUnavailable(container: HTMLElement, screen: BackendStartupScreen) {
  if (activeTab) {
    activeTab.destroy();
    activeTab = null;
  }
  const panel = renderEmptyPanel(container);
  renderedUnavailable = { screen, retryDisabled: settingsLoadInFlight };

  const starting = screen === "starting";

  const title = document.createElement("h2");
  title.className = "panel-title";
  title.textContent = starting ? "Starting JustSay…" : "Cannot load settings";

  const explanation = document.createElement("p");
  explanation.className = "panel-subtitle";
  explanation.textContent = starting
    ? "Waiting for the backend to start."
    : settingsError ?? BACKEND_STILL_STARTING_MESSAGE;

  panel.append(title, explanation);

  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "btn";
  retry.id = "btn-retry-settings";
  retry.textContent = "Try again";
  retry.disabled = settingsLoadInFlight;
  retry.addEventListener("click", () => {
    retry.disabled = true;
    retry.textContent = "Retrying…";
    void loadSettingsIntoUi();
  });
  panel.append(retry);
}

/** Empty the pane and put one `.panel` in it, scrolled to the top. */
function renderEmptyPanel(container: HTMLElement): HTMLElement {
  container.innerHTML = "";
  container.scrollTop = 0;
  const panel = document.createElement("div");
  panel.className = "panel";
  container.append(panel);
  return panel;
}

/** Whether the backend answered this request and refused it — a 401, or any
 *  other status it sent. A dropped socket, an abort and an expired budget are
 *  not answers, so they leave the window free to try again (ADR 092). */
function backendAnsweredAndFailed(error: unknown): boolean {
  return error instanceof ApiAuthError || error instanceof ApiRequestError;
}

/** Load the settings into the window, or paint the screen that says why not.
 *
 *  Serialized on `settingsLoadInFlight`, so two overlapping loads cannot finish
 *  out of order and have the loser's repaint erase the winner's tab.
 *  `observedReachable` is the caller's own `/health` reading, or null to take
 *  one here (ADR 092). */
async function loadSettingsIntoUi(observedReachable: boolean | null = null): Promise<void> {
  if (settingsLoadInFlight) return;
  settingsLoadInFlight = true;
  let reachable = false;
  try {
    reachable = observedReachable ?? (await probeBackend());
    await withTimeout(loadSettings(), BACKEND_WAIT_BUDGET_MS);
    settingsError = null;
    settingsLoadInFlight = false;
    switchPanel(currentPanel);
    applyProbeSchedule(currentStartupScreen());
  } catch (e) {
    const answered = backendAnsweredAndFailed(e);
    if (answered) answeredLoadFailures += 1;
    settingsError = settingsUnavailableMessage(e, reachable || answered);
    settingsLoadInFlight = false;
    const screen = currentStartupScreen();
    renderSettingsUnavailable(pane, screen);
    renderBackendStatus(backendReachable);
    applyProbeSchedule(screen);
    console.error("Failed to load settings:", e);
  }
}

/** The screen this window should be showing right now, and whether it should
 *  start a load (ADR 092). */
function currentStartupDecision(): BackendStartupDecision {
  return nextBackendStartup({
    loaded: settings !== null,
    loadInFlight: settingsLoadInFlight,
    backendAnswering: backendReachable,
    answeredFailures: answeredLoadFailures,
    msWaiting: performance.now() - startupWaitStartedAt,
  });
}

function currentStartupScreen(): BackendStartupScreen {
  return currentStartupDecision().screen;
}

/** Act on that decision: start the load it calls for, put the window on the
 *  screen it names, and schedule the poll accordingly.
 *
 *  The load starts before the repaint, so **Try again** is never painted live
 *  in front of a request that is already in flight. */
function applyStartupDecision() {
  const decision = currentStartupDecision();
  if (decision.load) void loadSettingsIntoUi(backendReachable);
  if (decision.screen !== "ready") refreshUnavailableScreen(decision.screen);
  applyProbeSchedule(decision.screen);
}

/** Repaint the waiting or failure screen only when what it would show has
 *  changed, so a poll does not rebuild a screen that already says this. */
function refreshUnavailableScreen(screen: BackendStartupScreen) {
  if (
    renderedUnavailable?.screen === screen &&
    renderedUnavailable.retryDisabled === settingsLoadInFlight
  ) {
    return;
  }
  renderSettingsUnavailable(pane, screen);
}

function switchPanel(panelName: PanelName) {
  if (activeTab) {
    activeTab.destroy();
    activeTab = null;
  }

  currentPanel = panelName;
  renderPanelSelection(sidebar, panelName);

  if (!settings) {
    renderSettingsUnavailable(pane, currentStartupScreen());
    return;
  }

  const panel = renderEmptyPanel(pane);
  activeTab = asLifecycle(panels[panelName](panel, settings, settingsWindowHidden));
  if (settingsWindowHidden) activeTab?.releaseResources?.();
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
  void applyAppTheme(loaded.theme);
  renderAccountName();
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
  const state = backendStateOf({
    reachable,
    refusedRequest: sawAuthFailure(),
    starting: currentStartupScreen() === "starting",
  });
  const diagnosis = state === "unauthorized"
    ? `Backend rejected an authenticated request (401). Tauri bridge: ${bridgeDiagnosisText(lastBridgeDiagnosis())}`
    : null;
  renderSidebarStatus(sidebarStatus, state, diagnosis);
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
  applyStartupDecision();
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


function renderAccountName() {
  renderAccountRow(
    sidebar.querySelector(".account-row")!,
    displayName(settings?.display_name ?? "", osAccountName),
  );
}

async function renameUser(chosen: string): Promise<void> {
  await saveSettings({ display_name: chosen });
  renderAccountName();
}

async function initAccountName() {
  osAccountName = await readOsDisplayName();
  renderAccountName();
  if (currentPanel === "account" && settings) switchPanel("account");
}


/** Open the panel another window asked for, scrolled to the section it names
 *  once the panel is drawn. */
function navigateTo(target: NavigatePanel) {
  if (target.panel !== currentPanel || !settings) switchPanel(target.panel);
  if (target.section === "meetings") {
    pane.querySelector("#meetings-toggle")?.closest(".card")?.scrollIntoView({ block: "center" });
  }
}

/** A language changed from the ring reaches this window's copy of the
 *  settings, and a Dictation panel on screen is drawn again to show it. */
async function followSettingsChangedElsewhere() {
  if (!settings || settingsLoadInFlight) return;
  const shownLanguage = settings.language;
  try {
    settings = await api.getSettings();
  } catch (e) {
    console.warn("Could not re-read the settings after a change elsewhere:", e);
    return;
  }
  if (settings.language !== shownLanguage && currentPanel === "dictation") switchPanel("dictation");
}

async function listenToOtherWindows() {
  try {
    const { listen } = await loadEventApi();
    await listen<NavigatePanel>(EVENT_NAVIGATE_PANEL, ({ payload }) => navigateTo(payload));
    await listen(EVENT_SETTINGS_CHANGED, () => void followSettingsChangedElsewhere());
  } catch {
  }
}


sidebar.querySelectorAll<HTMLButtonElement>("[data-panel]").forEach((item) => {
  item.addEventListener("click", () => {
    const panel = item.dataset.panel as PanelName;
    if (panel !== currentPanel || !settings) switchPanel(panel);
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
    applyStartupDecision();
    activeTab?.releaseResources?.();
    return;
  }
  void probeBackend();
  applyStartupDecision();
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
    const { listen } = await loadEventApi();
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


/** Hand the title bar's controls to this window; served by Vite there is no
 *  window behind the page, and the bar stays drawn without them. */
async function connectTitlebarToWindow() {
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await wireTitlebar(titlebar, getCurrentWindow());
  } catch {
  }
}

function init() {
  applyThemePreference("system");
  mountIconSprite(document);
  renderTitlebar(titlebar, detectShortcutPlatform(navigator));
  void connectTitlebarToWindow();
  void initAppVersion();
  void initAccountName();
  void trackTabWindowVisibility();
  void listenToOtherWindows();
  renderSettingsUnavailable(pane, currentStartupScreen());
  void probeBackend();
  applyStartupDecision();
}

init();
