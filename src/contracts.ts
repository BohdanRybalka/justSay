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
