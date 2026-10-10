/**
 * Values this TypeScript program must spell the same way as somebody else.
 *
 * Entry rule: a value belongs here only if some party outside this TypeScript
 * program writes it down too — the Python backend, the Rust shell, or another
 * WebView window on the Tauri event bus — and nothing at compile time checks
 * that the two spellings agree.
 *
 * The exclusion test, applied in one step: if every party that writes this
 * value down is compiled together with this file, it does not belong here. A
 * display string only one module renders, or a divisor only one module uses,
 * stays next to that module — `tsc` already keeps those honest, and this file
 * is the half of a contract nothing else can check, not a bag of constants.
 *
 * `backend/tests/test_cross_language_contracts.py` asserts each value here
 * agrees with its counterpart, and ADR 045 records why the copies are pinned
 * rather than generated.
 */

import type { PanelName } from "./settings/shell/sidebar";

export const BACKEND_PORT = 9377;

export const BACKEND_BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;

/** The sentinel the backend substitutes for a stored cloud key. Sending it
 *  back on a PUT is a no-op, so it doubles as "leave this key alone". */
export const MASKED_API_KEY = "***";

export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

export const ACCEPTED_AUDIO_EXTENSIONS: readonly string[] = [
  ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac",
  ".m4a", ".mp4", ".aac", ".opus", ".wma", ".aiff", ".aif",
];

export const EVENT_SETTINGS_CHANGED = "settings-changed";
export const EVENT_SHORTCUT_REQUESTED = "shortcut-requested";
export const EVENT_SHORTCUT_APPLIED = "shortcut-applied";
export const EVENT_MEETING_TOGGLE = "meeting-toggle";

/** The pointer entered or left the zone around the widget's pill. The shell
 *  reads the cursor itself: the zone reaches past the pill, and the page gets
 *  no mouse events where its window lets clicks through. */
export const EVENT_WIDGET_HOVER = "widget-hover";

/** The ring opened at the cursor, or closed. The shell shows and hides the
 *  ring's window itself, whatever closed it, and says so, so the page replays
 *  the fan-out on each opening and starts the next one from rest. */
export const EVENT_RING_OPENED = "ring-opened";
export const EVENT_RING_CLOSED = "ring-closed";

/** Where the pointer is while the ring is open. The shell reads the cursor
 *  itself: a webview in a window of an app that is not active is not reliably
 *  told where the mouse is, and the ring never activates the app. */
export const EVENT_RING_POINTER = "ring-pointer";

/** Another window asks the main window to open one of its panels. The asker
 *  shows the window itself. */
export const EVENT_NAVIGATE_PANEL = "navigate-panel";

/** The user picked an audio file in the system dialog the ring or the tray
 *  opened. The shell has already shown the main window; the payload names the
 *  file and the token that fetches its bytes (ADR 087). */
export const EVENT_FILE_PICKED = "file-picked";

/** The Settings window was dismissed. The shell intercepts `CloseRequested`,
 *  prevents it and hides the window, so the webview stays mounted and no tab's
 *  teardown ever runs — a microphone the Dictation panel holds outlives the window
 *  that opened it. The Rust side emits this after `hide()` and `settings.ts`
 *  calls the active tab's `releaseResources` on it, leaving the tab mounted.
 *  An event we emit ourselves rather than `visibilitychange`, which WebView2
 *  and WKWebView are not verifiably agreed on for a native hide. */
export const EVENT_SETTINGS_HIDDEN = "settings-hidden";

/** The Settings window came back. The shell emits it from the single helper
 *  every show path calls, so a tab that gave something up on the hide can take
 *  it back (ADR 089). It arrives on every show, including one where the window
 *  was already visible, and `settings.ts` is the one place that decides what a
 *  repeat means. */
export const EVENT_SETTINGS_SHOWN = "settings-shown";

/** What a `session_id` may look like on the wire, spelled the same way as
 *  `SESSION_ID_PATTERN` in `backend/app/audio/session.py` and pinned against
 *  it by `backend/tests/test_cross_language_contracts.py`. The backend answers
 *  422 to anything else, so a mint that drifted from this would fail every
 *  recording rather than one. */
export const SESSION_ID_PATTERN = /^[0-9a-f]{32}$/;

/** Every reason a meeting capture can report going wrong, spelled the same way
 *  as `CaptureIncident` in `backend/app/audio/meeting_recorder.py` and pinned
 *  against it by `backend/tests/test_cross_language_contracts.py`. The widget
 *  picks the sentence it shows off these tokens, so one the backend can send
 *  and this list does not carry would degrade the marker with no explanation. */
export const CAPTURE_INCIDENTS = [
  "microphone_stalled",
  "system_audio_ended",
  "storage_failed",
  "storage_low",
  "storage_backlog",
] as const;

export type CaptureIncident = (typeof CAPTURE_INCIDENTS)[number];

/** Payload of `EVENT_SHORTCUT_REQUESTED` — Settings asks the widget, which
 *  owns the global-shortcut registration, to take a new accelerator. */
export interface ShortcutRequested {
  shortcut: string;
}

/** Payload of `EVENT_WIDGET_HOVER`. */
export interface WidgetHover {
  inside: boolean;
}

/** Payload of `EVENT_NAVIGATE_PANEL`. */
export interface NavigatePanel {
  panel: PanelName;
}

/** Payload of `EVENT_RING_POINTER`: logical pixels from the ring's centre. */
export interface RingPointer {
  x: number;
  y: number;
}

/** Payload of `EVENT_SHORTCUT_APPLIED` — the widget's answer, carrying both
 *  whether the accelerator registered and whether it reached disk. */
export interface ShortcutApplied {
  shortcut: string;
  ok: boolean;
  reason: string | null;
  persisted: boolean | null;
  stillActive: string | null;
}
