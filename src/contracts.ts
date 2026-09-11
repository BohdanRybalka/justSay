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

export const BACKEND_PORT = 9377;

export const BACKEND_BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;

/** The sentinel the backend substitutes for a stored cloud key. Sending it
 *  back on a PUT is a no-op, so it doubles as "leave this key alone". */
export const MASKED_API_KEY = "***";

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export const ACCEPTED_AUDIO_EXTENSIONS: readonly string[] = [
  ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac",
  ".m4a", ".mp4", ".aac", ".opus", ".wma", ".aiff", ".aif",
];

export const EVENT_SETTINGS_CHANGED = "settings-changed";
export const EVENT_SHORTCUT_REQUESTED = "shortcut-requested";
export const EVENT_SHORTCUT_APPLIED = "shortcut-applied";
export const EVENT_MEETING_TOGGLE = "meeting-toggle";

/** The Settings window was dismissed. The shell intercepts `CloseRequested`,
 *  prevents it and hides the window, so the webview stays mounted and no tab's
 *  teardown ever runs — a microphone the General tab holds outlives the window
 *  that opened it. The Rust side emits this after `hide()` and `settings.ts`
 *  re-mounts the active tab on it, which releases whatever that tab held
 *  ([JS-121]). An event we emit ourselves rather than `visibilitychange`,
 *  which WebView2 and WKWebView are not verifiably agreed on for a native
 *  hide. */
export const EVENT_SETTINGS_HIDDEN = "settings-hidden";

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

/** Payload of `EVENT_SHORTCUT_APPLIED` — the widget's answer, carrying both
 *  whether the accelerator registered and whether it reached disk. */
export interface ShortcutApplied {
  shortcut: string;
  ok: boolean;
  reason: string | null;
  persisted: boolean | null;
  stillActive: string | null;
}
