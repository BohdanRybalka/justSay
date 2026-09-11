import {
  DEFAULT_SHORTCUT,
  detectShortcutPlatform,
  formatAccelerator,
  shortcutFailureMessage,
  shouldReapplyShortcut,
} from "../accelerator";
import { api, REQUEST_TIMEOUT_MS } from "../api";
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
import { newSessionId } from "../session";
import { isStaleStatusResponse } from "../stale-response";
import { TimedOutError, withTimeout } from "../timeout";
import { createAbandonedSessions } from "./abandoned-request";
import { computeDoneStatus } from "./done-status";
import {
  DICTATION_NEVER_PROCESSED,
  dictationErrorLabel,
  startErrorLabel,
  type DictationErrorLabel,
} from "./error-label";
import { decideMeetingHealth } from "./meeting-health";
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
let autoRevertTimer: ReturnType<typeof setTimeout> | null = null;
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

  if (autoRevertTimer) {
    clearTimeout(autoRevertTimer);
    autoRevertTimer = null;
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
      autoRevertTimer = setTimeout(() => {
        autoRevertTimer = null;
        if (state === "done") setState("idle");
      }, AUTO_REVERT_MS);
      break;
    case "error":
      text.textContent = message || "Error";
      durationEl.textContent = "";
      renderIcon("error");
      autoRevertTimer = setTimeout(() => {
        autoRevertTimer = null;
        if (state === "error") setState("idle");
      }, AUTO_REVERT_MS);
      break;
  }
}

/** `start` exists so an adopted recording shows the backend's elapsed time
 *  rather than restarting the stopwatch at zero: when a start times out and the
 *  backend turns out to be holding this window's own session, the capture began
 *  before the budget ran out.
 *
 *  It clears the interval it is about to replace, so the one function that
 *  creates the stopwatch is also the one that owns there being only one of it:
 *  the adoption path calls this while `setState("recording")` has already armed
 *  an interval, and two of them would write to the same node. */
function startDurationTimer(start = Date.now()) {
  if (durationInterval) clearInterval(durationInterval);
  const update = () => {
    const elapsed = (Date.now() - start) / 1000;
    durationEl.textContent = formatStopwatch(elapsed);
  };
  update();
  durationInterval = setInterval(update, 100);
}


/** The sessions this window started and cannot account for.
 *
 *  Discharged on the connection poll, which is the only thing that keeps
 *  running while the backend is unreachable — the timeout site itself has just
 *  proved the backend is not answering, so a probe sent there would go into the
 *  same silence. */
const abandoned = createAbandonedSessions({
  discard: (sessionId) => api.audioDiscard(sessionId),
});

/** The session backing the dictation currently on screen. */
let currentSession = "";

async function startRecording() {
  if (state === "recording" || state === "processing") return;

  const session = newSessionId();
  currentSession = session;
  setState("recording");

  const startIssuedAt = Date.now();
  try {
    await api.audioStart(session);
  } catch (e) {
    if (e instanceof TimedOutError && (await adoptTimedOutStart(session, startIssuedAt))) {
      console.warn("Start recording timed out but the backend holds this session; adopted it", e);
      return;
    }
    reportTransitionFailure(startErrorLabel(e));
    console.error("Start recording failed:", e);
  }
}

/** Decide what an abandoned start actually did, from the id it carried.

 *  One rule, because there is exactly one thing the backend can prove here: it
 *  is holding this window's own session, so the capture is running and is
 *  adopted with the stopwatch continuing from the reported elapsed time.
 *
 *  Every other answer proves nothing, so the session is owed. Nothing
 *  serializes the queued `POST /audio/start` against the rest of the event
 *  loop: an idle recorder can still be taken by it a moment later, and a
 *  recorder held by somebody else can be released by that owner's stop before
 *  the handler runs, which leaves the same device open under a name the app no
 *  longer holds. A status read that failed proves less still. What settles the
 *  debt is the discard's own answer — 200 if the start did land, 403 or 409 if
 *  it never did — rather than a status read taken before the handler ran.
 *
 *  The debt waits `REQUEST_TIMEOUT_MS` from the instant the start was *issued*,
 *  which is the same eligibility wait a dictation gets and is measured from the
 *  same place. Anchoring it to the moment adoption gave up instead would stack
 *  the status read's own budget on top of the wait — a start abandoned at 15 s
 *  and a status read abandoned at 15 s would hold the probe back for 45 s — and
 *  the invariant the wait exists for is about the queued handler, which was
 *  queued when the start went out. The probe still cannot beat that handler to
 *  the recorder's lock and then read its own earliness as proof.
 *
 *  A start that failed with an *answer* — a refused connection, a 409, a 422 —
 *  never reaches this function, and must not: those are decisive on their own
 *  and owing them would turn a settled failure into debt. */
async function adoptTimedOutStart(session: string, startIssuedAt: number): Promise<boolean> {
  const status = await api.audioStatus().catch(() => null);

  if (status?.is_recording && status.session_id === session) {
    startDurationTimer(Date.now() - status.duration_seconds * 1000);
    return true;
  }

  void abandoned.owe(session, startIssuedAt + REQUEST_TIMEOUT_MS);
  return false;
}

/** Reports a failed start or a failed dictation. The label is the caller's —
 *  `startErrorLabel` and `dictationErrorLabel` say different things and must
 *  keep doing so — and the pill and the toast are raised together here so a
 *  caller cannot raise one without the other. */
function reportTransitionFailure({ label, toast }: DictationErrorLabel) {
  setState("error", label);
  notifyError(toast);
}

/** Stop, transcribe, and stop waiting once the dictation is proved dead.
 *
 *  `/pipeline/dictate` has no budget and cannot honestly be given one — a local
 *  transcription is legitimately slow and no measured upper bound for it exists
 *  — so before spec 119 an accepted-and-never-answered dictate parked the
 *  widget in `processing` for the 600 s budget with the recorder still
 *  appending frames the whole time.
 *
 *  The session id answers the question that budget was dodging: *has this
 *  request been processed at all?* The dictate handler's first act is
 *  `recorder.stop()`, so a backend still holding this session has not run it.
 *  The obligation is recorded before the request goes out and raced against it;
 *  the discard that resolves it is not eligible until `REQUEST_TIMEOUT_MS` has
 *  passed, which is what stops a probe beating a healthy handler to the lock.
 *  Worst case in `processing`: that wait, plus one 5 s poll period, plus the
 *  probe's own 15 s budget.
 *
 *  A probe that came back `not-live` is not that outcome: `403` and `409` are
 *  the recorder saying it is no longer holding this session, which is what a
 *  dictate handler that already ran leaves behind. The widget then goes back to
 *  waiting for that dictation rather than declaring it dead. */
async function stopAndProcess() {
  if (state !== "recording") return;

  const session = currentSession;
  setState("processing");

  const neverProcessed = abandoned.owe(session, Date.now() + REQUEST_TIMEOUT_MS);
  const dictated = api.dictate(session, currentLanguage).then(
    (result) => ({ kind: "answered", result }) as const,
    (error) => ({ kind: "failed", error }) as const,
  );

  const raced = await Promise.race([
    dictated,
    neverProcessed.then((proof) => ({ kind: "proof", proof }) as const),
  ]);

  const outcome =
    raced.kind !== "proof" ? raced
    : raced.proof === "proven" ? ({ kind: "never-processed" } as const)
    : await dictated;

  if (outcome.kind === "never-processed") {
    reportTransitionFailure(DICTATION_NEVER_PROCESSED);
    return;
  }

  abandoned.forget(session);

  if (outcome.kind === "failed") {
    reportTransitionFailure(dictationErrorLabel(outcome.error));
    console.error("Pipeline failed:", outcome.error);
    return;
  }

  const status = computeDoneStatus(outcome.result);
  if (!status) {
    setState("idle");
    return;
  }
  setState("done", status.label, formatStopwatch(status.elapsedSeconds));
  if (outcome.result.discarded_reason !== "silence") {
    renderRouteBadge(outcome.result);
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
let meetingIncident: string | null = null;
let meetingTicks = 0;

const MEETING_TICK_MS = 500;

/** The backend is asked how the capture is doing on every fourth clock tick,
 *  so the poll runs every 2 s and no second timer appears. A capture that dies
 *  mid-meeting used to leave a ticking indicator over nothing for the rest of
 *  the call, because `syncMeetingIndicator` is awaited once at load and
 *  nothing repeated it. */
const MEETING_POLL_EVERY_TICKS = 4;

function renderMeetingIndicatorFromState() {
  renderMeetingIndicator(widget, {
    active: meetingActive,
    elapsedSeconds: (Date.now() - meetingStartedAt) / 1000,
    incident: meetingIncident,
  });
}

function onMeetingTick() {
  renderMeetingIndicatorFromState();
  meetingTicks += 1;
  if (meetingTicks % MEETING_POLL_EVERY_TICKS === 0) void pollMeetingHealth();
}

function beginMeetingIndicator(startedAt = Date.now()) {
  meetingActive = true;
  meetingStartedAt = startedAt;
  meetingIncident = null;
  meetingTicks = 0;
  renderMeetingIndicatorFromState();
  meetingTimer = setInterval(onMeetingTick, MEETING_TICK_MS);
}

function endMeetingIndicator() {
  meetingActive = false;
  meetingIncident = null;
  if (meetingTimer) {
    clearInterval(meetingTimer);
    meetingTimer = null;
  }
  renderMeetingIndicatorFromState();
}

/** Act on one poll of the meeting status. A failed poll is ignored on purpose:
 *  an unreachable backend is not evidence the capture ended, and taking the
 *  marker down on it would break ADR 040 obligation 2 while a recording may
 *  still be running. */
async function pollMeetingHealth() {
  let status;
  try {
    status = await api.getMeetingStatus();
  } catch (e) {
    console.warn("Could not read the meeting recording status:", e);
    return;
  }
  const action = decideMeetingHealth(status, meetingActive);
  if (action.kind === "keep") return;
  if (action.kind === "end") {
    endMeetingIndicator();
    await invokeShell("set_meeting_recording", { active: false });
    meetingToggleActions.reportError(action.message);
    return;
  }
  if (meetingIncident === action.incident) return;
  meetingIncident = action.incident;
  renderMeetingIndicatorFromState();
  meetingToggleActions.reportError(action.message);
}

/** The budget on a shell command, covering the bridge import and the `invoke()`
 *  behind it as one unit.
 *
 *  Neither step has a bound of its own — a dynamic import has no timeout and
 *  `invoke()` has no reject channel at all (ADR 028) — so an absent bridge
 *  leaves this promise pending for the life of the window. The meeting toggle
 *  awaits it while holding `meetingBusy`, which would swallow every later
 *  press, tray included: the wedge ADR 049 gave the HTTP calls a budget to
 *  remove, one layer down. Matched to `api.ts`'s own bridge budget, because it
 *  is the same two steps against the same transport. */
const SHELL_INVOKE_TIMEOUT_MS = 3000;

async function invokeShell(command: string, args?: Record<string, unknown>) {
  try {
    await withTimeout(
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke(command, args);
      })(),
      SHELL_INVOKE_TIMEOUT_MS,
    );
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


let latestConnectionProbeToken = 0;

/** `setInterval` does not await this function, so against a backend that
 *  accepts and abandons, a probe outlives the 5 s interval and several are in
 *  flight at once — each then writing `connectionState` from whatever it read
 *  when it started, in fetch-completion order rather than start order, which
 *  duplicates or swallows the unreachable toast.
 *
 *  The generation counter discards the superseded *answers* and keeps every
 *  probe, which an in-flight boolean cannot: a boolean skips a tick outright,
 *  so with a 15 s budget on a 5 s poll the widget can sit a whole budget past a
 *  tick before it notices the backend came back. Same guard, same shape, as the
 *  Settings window's badge and the Models tab's status read. */
async function checkConnection() {
  void abandoned.settle(Date.now());

  const token = ++latestConnectionProbeToken;
  let healthOk = true;
  try {
    await api.health();
  } catch {
    healthOk = false;
  }
  if (isStaleStatusResponse(token, latestConnectionProbeToken)) return;
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
