import { api } from "../api";
import { notifyError, nextConnectionCheckState, type ConnectionCheckState } from "../notify";
import { computeDoneStatus } from "./done-status";

// --- State ---

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
let isTransitioning = false;
let isHovered = false;
let durationInterval: ReturnType<typeof setInterval> | null = null;
let iconFlashTimer: ReturnType<typeof setTimeout> | null = null;
let connectionState: ConnectionCheckState = { offline: false, firstCheckDone: false };

// Settings (loaded from backend)
let currentShortcut = "Ctrl+Alt+KeyV";
let currentLanguage = "uk";

const AUTO_REVERT_MS = 3000;

// --- DOM ---

const widget = document.getElementById("widget")!;
const iconEl = document.getElementById("widget-icon")!;
const text = document.getElementById("widget-text")!;
const durationEl = document.getElementById("widget-duration")!;

// --- Icon helpers ---

function updateIcon(next: IconState) {
  const keep = [...iconEl.classList].filter(
    (c) =>
      c.startsWith("js-widget--") &&
      !ICON_STATE_MODIFIERS.has(c.slice("js-widget--".length) as IconState),
  );
  iconEl.className = ["widget-icon", "js-widget", ...keep, `js-widget--${next}`].join(" ");
}

// "Interactive window" — when hover should respond. Idle is the obvious case;
// the calm tail of `done` (after the 700 ms one-shot, while the compact status
// is still showing) is also interactive, so the icon doesn't get stuck on `hover`.
function isInteractive(): boolean {
  return state === "idle" || (state === "done" && iconFlashTimer === null);
}

// --- State management ---

function setState(newState: WidgetState, message?: string, durationLabel?: string) {
  state = newState;
  widget.className = `widget ${state}`;

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
      updateIcon(isHovered ? "hover" : "idle");
      break;
    case "recording":
      text.textContent = "Recording";
      startDurationTimer();
      updateIcon("recording");
      break;
    case "processing":
      text.textContent = "Processing";
      durationEl.textContent = "";
      updateIcon("processing");
      break;
    case "done":
      text.textContent = message || "Done";
      durationEl.textContent = durationLabel || "";
      updateIcon("done");
      iconFlashTimer = setTimeout(() => {
        iconFlashTimer = null;
        if (state === "done") updateIcon(isHovered ? "hover" : "idle");
      }, 700);
      setTimeout(() => {
        if (state === "done") setState("idle");
      }, AUTO_REVERT_MS);
      break;
    case "error":
      text.textContent = message || "Error";
      durationEl.textContent = "";
      updateIcon("error");
      setTimeout(() => {
        if (state === "error") setState("idle");
      }, AUTO_REVERT_MS);
      break;
  }
}

function startDurationTimer() {
  const start = Date.now();
  const update = () => {
    const elapsed = (Date.now() - start) / 1000;
    durationEl.textContent = formatDuration(elapsed);
  };
  update();
  durationInterval = setInterval(update, 100);
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  if (m > 0) return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
  return `${s}.${ms}s`;
}

// --- Recording flow ---

async function startRecording() {
  if (state === "recording" || state === "processing" || isTransitioning) return;

  isTransitioning = true;
  setState("recording");

  try {
    await api.audioStart();
  } catch (e) {
    setState("error", "Start failed");
    notifyError("Couldn't start recording — try again.");
    console.error("Start recording failed:", e);
  } finally {
    isTransitioning = false;
  }
}

async function stopAndProcess() {
  if (state !== "recording" || isTransitioning) return;

  isTransitioning = true;
  setState("processing");

  try {
    const result = await api.dictate(currentLanguage);
    const outcome = computeDoneStatus(result);
    if (outcome) {
      setState("done", outcome.label, formatDuration(outcome.elapsedSeconds));
      showRouteBadge(result);
    } else {
      setState("idle");
    }
  } catch (e) {
    const msg = (e instanceof Error ? e.message : String(e)).toLowerCase();
    const errorLabel = msg.includes("missing") ? "Add key in Settings" : "Failed";
    setState("error", errorLabel);
    notifyError(
      msg.includes("missing")
        ? "No API key set — add one in Settings."
        : "Dictation failed — try again.",
    );
    console.error("Pipeline failed:", e);
  } finally {
    isTransitioning = false;
  }
}

function showRouteBadge(result: { model_name?: string; duration_ms: number; fallback_reason?: string | null }) {
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

async function toggleRecording() {
  if (isTransitioning) return;

  if (state === "recording") {
    await stopAndProcess();
  } else if (state === "idle" || state === "done" || state === "error") {
    await startRecording();
  }
}

// --- Click ---

widget.addEventListener("click", () => {
  toggleRecording();
});

// --- Hover (icon-only — only flips when the widget is in an interactive state) ---

widget.addEventListener("mouseenter", () => {
  isHovered = true;
  if (isInteractive()) updateIcon("hover");
});

widget.addEventListener("mouseleave", () => {
  isHovered = false;
  if (isInteractive()) updateIcon("idle");
});

// --- Global shortcut ---

let unregisterFn: (() => Promise<void>) | null = null;

async function setupGlobalShortcut(shortcut: string) {
  try {
    const { register, unregister } = await import("@tauri-apps/plugin-global-shortcut");

    // Unregister previous if exists
    if (unregisterFn) {
      try {
        await unregisterFn();
      } catch { /* ignore */ }
    }

    await register(shortcut, (event) => {
      if (event.state === "Pressed") {
        startRecording();
      } else if (event.state === "Released") {
        stopAndProcess();
      }
    });

    unregisterFn = () => unregister(shortcut);
    console.log(`Global shortcut registered: ${shortcut}`);
  } catch (e) {
    console.warn("Global shortcut not available:", e);
  }
}

// --- Load settings ---

async function loadSettings() {
  try {
    const settings = await api.getSettings();
    currentLanguage = settings.language;
    if (settings.shortcut !== currentShortcut) {
      currentShortcut = settings.shortcut;
      await setupGlobalShortcut(currentShortcut);
    }
  } catch (e) {
    console.warn("Failed to load settings:", e);
  }
}

// --- Listen for settings changes from Settings window ---

async function listenForSettingsChanges() {
  try {
    const { listen } = await import("@tauri-apps/api/event");
    await listen("settings-changed", async () => {
      await loadSettings();
    });
  } catch {
    // Not in Tauri
  }
}

// --- Health check ---

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
  } else {
    if (state === "idle") text.textContent = "Offline";
    if (result.shouldNotify) notifyError("JustSay backend is unreachable.");
  }
}

// --- Init ---

async function init() {
  await checkConnection();
  setInterval(checkConnection, 5000);

  await loadSettings();
  await setupGlobalShortcut(currentShortcut);
  await listenForSettingsChanges();

  // Signal Rust that webview is loaded
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("widget_ready");
  } catch {
    // Not in Tauri
  }
}

init();
