import {
  DEFAULT_SHORTCUT,
  detectShortcutPlatform,
  formatAccelerator,
  shortcutFailureMessage,
  shouldReapplyShortcut,
} from "../accelerator";
import { api } from "../api";
import {
  EVENT_MEETING_TOGGLE,
  EVENT_SETTINGS_CHANGED,
  EVENT_SHORTCUT_APPLIED,
  EVENT_SHORTCUT_REQUESTED,
  type ShortcutApplied,
  type ShortcutRequested,
} from "../contracts";
import { formatStopwatch } from "../format";
import { notifyError, nextConnectionCheckState, type ConnectionCheckState } from "../notify";
import { computeDoneStatus } from "./done-status";
import { dictationErrorLabel } from "./error-label";
import { MEETING_STATE_CLASS, renderMeetingIndicator } from "./meeting-indicator";
import { type MeetingToggleActions, runMeetingToggle } from "./meeting-toggle";
import { createRecordingIntentQueue } from "./recording-intent";
import { CONNECTION_POLL_MS, createSettingsRetry } from "./settings-retry";


type WidgetState = "idle" | "recording" | "processing" | "done" | "error";
type IconState = "idle" | "hover" | "recording" | "processing" | "done" | "error";

const ICON_STATE_MODIFIERS: ReadonlySet<IconState> = new Set<IconState>([
  "idle",
  "hover",
  "recording",
  "processing",
  "done",
  "error",
]);

let state: WidgetState = "idle";
let isHovered = false;
let durationInterval: ReturnType<typeof setInterval> | null = null;
let iconFlashTimer: ReturnType<typeof setTimeout> | null = null;
let connectionState: ConnectionCheckState = { offline: false, firstCheckDone: false };

let currentShortcut = DEFAULT_SHORTCUT;
let currentLanguage = "uk";
const shortcutPlatform = detectShortcutPlatform(navigator);

const AUTO_REVERT_MS = 3000;


const widget = document.getElementById("widget")!;
const iconEl = document.getElementById("widget-icon")!;
const text = document.getElementById("widget-text")!;
const durationEl = document.getElementById("widget-duration")!;


function renderIcon(next: IconState) {
  const keep = [...iconEl.classList].filter(
    (c) =>
      c.startsWith("js-widget--") &&
      !ICON_STATE_MODIFIERS.has(c.slice("js-widget--".length) as IconState),
  );
  iconEl.className = ["widget-icon", "js-widget", ...keep, `js-widget--${next}`].join(" ");
}

function isInteractive(): boolean {
  return state === "idle" || (state === "done" && iconFlashTimer === null);
}


function setState(newState: WidgetState, message?: string, durationLabel?: string) {
  state = newState;
  widget.className = `widget ${state}${meetingActive ? ` ${MEETING_STATE_CLASS}` : ""}`;

  if (durationInterval && state !== "recording") {
    clearInterval(durationInterval);
    durationInterval = null;
  }

  if (iconFlashTimer) {
    clearTimeout(iconFlashTimer);
    iconFlashTimer = null;
  }

  switch (state) {
    case "idle":
      text.textContent = "JustSay";
      durationEl.textContent = "";
      renderIcon(isHovered ? "hover" : "idle");
      break;
    case "recording":
      text.textContent = "Recording";
      startDurationTimer();
      renderIcon("recording");
      break;
    case "processing":
      text.textContent = "Processing";
      durationEl.textContent = "";
      renderIcon("processing");
      break;
    case "done":
      text.textContent = message || "Done";
      durationEl.textContent = durationLabel || "";
      renderIcon("done");
      iconFlashTimer = setTimeout(() => {
        iconFlashTimer = null;
        if (state === "done") renderIcon(isHovered ? "hover" : "idle");
      }, 700);
      setTimeout(() => {
        if (state === "done") setState("idle");
      }, AUTO_REVERT_MS);
      break;
    case "error":
      text.textContent = message || "Error";
      durationEl.textContent = "";
      renderIcon("error");
      setTimeout(() => {
        if (state === "error") setState("idle");
      }, AUTO_REVERT_MS);
      break;
  }
}

/** Clears the interval it is about to replace, so the one function that creates
 *  the stopwatch is also the one that owns there being only one of it: two of
 *  them would write to the same node. */
function startDurationTimer() {
  const start = Date.now();
  if (durationInterval) clearInterval(durationInterval);
  const update = () => {
    const elapsed = (Date.now() - start) / 1000;
    durationEl.textContent = formatStopwatch(elapsed);
  };
  update();
  durationInterval = setInterval(update, 100);
}


async function startRecording() {
  if (state === "recording" || state === "processing") return;

  setState("recording");

  try {
    await api.audioStart();
  } catch (e) {
    setState("error", "Start failed");
    notifyError("Couldn't start recording — try again.");
    console.error("Start recording failed:", e);
  }
}

async function stopAndProcess() {
  if (state !== "recording") return;

  setState("processing");

  try {
    const result = await api.dictate(currentLanguage);
    const outcome = computeDoneStatus(result);
    if (outcome) {
      setState("done", outcome.label, formatStopwatch(outcome.elapsedSeconds));
      if (result.discarded_reason !== "silence") {
        renderRouteBadge(result);
      }
    } else {
      setState("idle");
    }
  } catch (e) {
    const { label, toast } = dictationErrorLabel(e);
    setState("error", label);
    notifyError(toast);
    console.error("Pipeline failed:", e);
  }
}

const recordingIntent = createRecordingIntentQueue({
  isRecording: () => state === "recording",
  isBusy: () => state === "processing",
  startRecording,
  stopRecording: stopAndProcess,
  reportError: (e) => console.error("Recording transition failed:", e),
});

function renderRouteBadge(result: { model_name?: string; duration_ms: number; fallback_reason?: string | null }) {
  const badge = document.getElementById("widget-route");
  if (!badge) return;

  const model = (result.model_name || "").split("/").pop() || "stt";
  const seconds = (result.duration_ms / 1000).toFixed(2);
  const fallback = result.fallback_reason ? " · fallback" : "";
  badge.textContent = `${model} · ${seconds} s${fallback}`;
  badge.classList.add("visible");
  if (result.fallback_reason) {
    badge.title = result.fallback_reason;
  } else {
    badge.removeAttribute("title");
  }
  setTimeout(() => badge.classList.remove("visible"), 4000);
}

let meetingActive = false;
let meetingStartedAt = 0;
let meetingTimer: ReturnType<typeof setInterval> | null = null;
let meetingBusy = false;

const MEETING_TICK_MS = 500;

function renderMeetingIndicatorFromState() {
  renderMeetingIndicator(widget, {
    active: meetingActive,
    elapsedSeconds: (Date.now() - meetingStartedAt) / 1000,
  });
}

function beginMeetingIndicator(startedAt = Date.now()) {
  meetingActive = true;
  meetingStartedAt = startedAt;
  renderMeetingIndicatorFromState();
  meetingTimer = setInterval(renderMeetingIndicatorFromState, MEETING_TICK_MS);
}

function endMeetingIndicator() {
  meetingActive = false;
  if (meetingTimer) {
    clearInterval(meetingTimer);
    meetingTimer = null;
  }
  renderMeetingIndicatorFromState();
}

async function invokeShell(command: string, args?: Record<string, unknown>) {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke(command, args);
  } catch (e) {
    console.warn(`Shell command ${command} failed:`, e);
  }
}

const meetingToggleActions: MeetingToggleActions = {
  isRecording: () => meetingActive,
  startRecording: () => api.startMeetingRecording(),
  stopRecording: () => api.stopMeetingRecording(),
  showIndicator: beginMeetingIndicator,
  hideIndicator: endMeetingIndicator,
  setTrayRecording: (active) => invokeShell("set_meeting_recording", { active }),
  openDisclosure: () => invokeShell("show_settings_window"),
  reportError: (message) => {
    console.error("Meeting recording:", message);
    notifyError(message);
  },
};

async function toggleMeetingRecording() {
  if (meetingBusy) return;
  meetingBusy = true;
  try {
    await runMeetingToggle(meetingToggleActions);
  } finally {
    meetingBusy = false;
  }
}

/** The widget window can be reloaded while a recording is running, and the
 *  indicator is the only thing telling the room a call is being captured — so
 *  it is restored from the backend rather than from this window's memory. */
async function syncMeetingIndicator() {
  try {
    const status = await api.getMeetingStatus();
    if (status.is_recording === meetingActive) return;
    if (status.is_recording) {
      beginMeetingIndicator(Date.now() - status.duration_seconds * 1000);
    } else {
      endMeetingIndicator();
    }
    await invokeShell("set_meeting_recording", { active: status.is_recording });
  } catch (e) {
    console.warn("Could not read the meeting recording status:", e);
  }
}

widget.addEventListener("click", () => {
  if (meetingActive) return;
  void recordingIntent.request("toggle");
});


widget.addEventListener("mouseenter", () => {
  isHovered = true;
  if (isInteractive()) renderIcon("hover");
});

widget.addEventListener("mouseleave", () => {
  isHovered = false;
  if (isInteractive()) renderIcon("idle");
});


type GlobalShortcutPlugin = typeof import("@tauri-apps/plugin-global-shortcut");
type ShortcutOutcome =
  | { ok: true; applied: boolean }
  | { ok: false; reason: string; stillActive: string | null };
type RequestedShortcutResult = {
  outcome: ShortcutOutcome;
  persisted: boolean | null;
  writeError: string | null;
};

let unregisterFn: (() => Promise<void>) | null = null;
let activeShortcut: string | null = null;
let shortcutFailureNotified: string | null = null;
let shortcutQueue: Promise<unknown> = Promise.resolve();

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function onShortcutEvent(event: { state: "Pressed" | "Released" }) {
  if (meetingActive && event.state === "Pressed") return;
  void recordingIntent.request(event.state === "Pressed" ? "start" : "stop");
}

async function releaseActiveShortcut() {
  if (unregisterFn) {
    try {
      await unregisterFn();
    } catch {}
  }
  unregisterFn = null;
  activeShortcut = null;
}

async function registerShortcut(
  plugin: GlobalShortcutPlugin,
  shortcut: string,
): Promise<{ ok: true } | { ok: false; reason: string }> {
  try {
    await plugin.register(shortcut, onShortcutEvent);
  } catch (e) {
    return { ok: false, reason: errorText(e) };
  }

  unregisterFn = () => plugin.unregister(shortcut);
  activeShortcut = shortcut;
  console.log(`Global shortcut registered: ${shortcut}`);
  return { ok: true };
}

async function runApplyShortcut(next: string, force: boolean): Promise<ShortcutOutcome> {
  try {
    const plugin = await import("@tauri-apps/plugin-global-shortcut");

    if (!force && !shouldReapplyShortcut(next, activeShortcut, await plugin.isRegistered(next))) {
      return { ok: true, applied: false };
    }

    const previous = activeShortcut;
    await releaseActiveShortcut();

    const attempt = await registerShortcut(plugin, next);
    if (attempt.ok) return { ok: true, applied: true };

    if (previous && previous !== next && (await registerShortcut(plugin, previous)).ok) {
      return { ok: false, reason: attempt.reason, stillActive: previous };
    }
    return { ok: false, reason: attempt.reason, stillActive: null };
  } catch (e) {
    return { ok: false, reason: errorText(e), stillActive: activeShortcut };
  }
}

function enqueueShortcutJob<T>(job: () => Promise<T>): Promise<T> {
  const queued = shortcutQueue.then(job);
  shortcutQueue = queued.catch(() => {});
  return queued;
}

function applyShortcut(next: string, options: { force: boolean }): Promise<ShortcutOutcome> {
  return enqueueShortcutJob(() => runApplyShortcut(next, options.force));
}

function reportShortcutOutcome(shortcut: string, outcome: ShortcutOutcome) {
  if (outcome.ok) {
    shortcutFailureNotified = null;
    widget.removeAttribute("title");
    return;
  }

  const message = shortcutFailureMessage(
    formatAccelerator(shortcut, shortcutPlatform),
    outcome.reason,
    outcome.stillActive ? formatAccelerator(outcome.stillActive, shortcutPlatform) : null,
  );
  widget.title = message;
  if (shortcutFailureNotified !== shortcut) {
    shortcutFailureNotified = shortcut;
    notifyError(message);
  }
}

async function announceShortcutOutcome(
  shortcut: string,
  outcome: ShortcutOutcome,
  persisted: boolean | null,
  writeError: string | null,
) {
  try {
    const { emit } = await import("@tauri-apps/api/event");
    const applied: ShortcutApplied = {
      shortcut,
      ok: outcome.ok,
      reason: outcome.ok ? writeError : outcome.reason,
      persisted,
      stillActive: outcome.ok ? shortcut : outcome.stillActive,
    };
    await emit(EVENT_SHORTCUT_APPLIED, applied);
  } catch {}
}

async function applyAndReportShortcut(shortcut: string) {
  const outcome = await applyShortcut(shortcut, { force: false });
  if (outcome.ok && !outcome.applied) return;
  reportShortcutOutcome(shortcut, outcome);
  await announceShortcutOutcome(shortcut, outcome, null, null);
}

async function runRequestedShortcut(shortcut: string): Promise<RequestedShortcutResult> {
  const outcome = await runApplyShortcut(shortcut, true);
  if (!outcome.ok) return { outcome, persisted: null, writeError: null };

  try {
    await api.updateSettings({ shortcut });
    currentShortcut = shortcut;
    return { outcome, persisted: true, writeError: null };
  } catch (e) {
    console.error("Failed to store the registered shortcut:", e);
    return { outcome, persisted: false, writeError: errorText(e) };
  }
}

async function applyRequestedShortcut(shortcut: string) {
  const { outcome, persisted, writeError } = await enqueueShortcutJob(() =>
    runRequestedShortcut(shortcut),
  );
  reportShortcutOutcome(shortcut, outcome);
  await announceShortcutOutcome(shortcut, outcome, persisted, writeError);
}


const settingsRetry = createSettingsRetry({
  now: () => performance.now(),
  isBusy: () => state === "recording" || state === "processing",
  fetchSettings: () => api.getSettings(),
  applySettings: async (settings) => {
    currentLanguage = settings.language;
    currentShortcut = settings.shortcut;
    await applyAndReportShortcut(currentShortcut);
  },
  applyFallbackShortcut: () => applyAndReportShortcut(currentShortcut),
  reportAttemptFailed: (e) => console.warn("Failed to load settings:", e),
  reportFallbackFailed: (e) => console.warn("Failed to apply the fallback shortcut:", e),
  reportGaveUp: () =>
    notifyError(
      "JustSay could not read your settings — the default language and shortcut are in use. " +
        "Restart the app to apply them.",
    ),
});


async function listenForSettingsChanges() {
  try {
    const { listen } = await import("@tauri-apps/api/event");
    await listen(EVENT_SETTINGS_CHANGED, async () => {
      await settingsRetry.load();
    });
    await listen<ShortcutRequested>(EVENT_SHORTCUT_REQUESTED, async ({ payload }) => {
      await applyRequestedShortcut(payload.shortcut);
    });
    await listen(EVENT_MEETING_TOGGLE, async () => {
      await toggleMeetingRecording();
    });
  } catch {
  }
}


async function checkConnection() {
  let healthOk = true;
  try {
    await api.health();
  } catch {
    healthOk = false;
  }
  const result = nextConnectionCheckState(connectionState, healthOk);
  connectionState = { offline: result.offline, firstCheckDone: result.firstCheckDone };

  if (healthOk) {
    if (state === "idle" && text.textContent === "Offline") text.textContent = "JustSay";
    await settingsRetry.retryIfDue();
  } else {
    if (state === "idle") text.textContent = "Offline";
    if (result.shouldNotify) notifyError("JustSay backend is unreachable.");
  }
}


async function init() {
  await checkConnection();
  setInterval(checkConnection, CONNECTION_POLL_MS);

  void settingsRetry.load();
  await listenForSettingsChanges();
  await syncMeetingIndicator();

  await invokeShell("widget_ready");
}

init();
