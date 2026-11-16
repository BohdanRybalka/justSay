/**
 * What each button in the column says and what picking it does, decided from
 * the settings and the meeting state the column last read. Nothing here talks
 * to a window or the backend; `ring.ts` carries out the step this module names.
 */

import { formatAccelerator, type ShortcutPlatform } from "../accelerator";
import type { UserSettings } from "../api";
import type { NavigatePanel } from "../contracts";
import { languageName } from "../languages";

export type RingAction = "meeting" | "file" | "dictate" | "language" | "settings";

export interface RingState {
  settings: Pick<
    UserSettings,
    "language" | "previous_language" | "meeting_consent_acknowledged" | "shortcut"
  > | null;
  meetingRecording: boolean;
}

/** A button's name, and the shortcut that does the same without the column. */
export interface ActionLabel {
  text: string;
  shortcut: string | null;
}

export type RingStep =
  | { kind: "toggle-meeting" }
  | { kind: "pick-file" }
  | { kind: "toggle-dictation" }
  | { kind: "switch-language"; language: string }
  | { kind: "open-panel"; target: NavigatePanel };

export const UNREAD_RING_STATE: RingState = { settings: null, meetingRecording: false };

export function actionLabel(action: RingAction, state: RingState, platform: ShortcutPlatform): ActionLabel {
  switch (action) {
    case "meeting":
      return named(state.meetingRecording ? "Stop the meeting" : "Record a meeting");
    case "file":
      return named("Transcribe a file");
    case "dictate":
      return {
        text: "Dictate",
        shortcut: state.settings ? formatAccelerator(state.settings.shortcut, platform) : null,
      };
    case "language":
      return named(state.settings ? `Language · ${languageName(state.settings.language)}` : "Language");
    case "settings":
      return named("Settings");
  }
}

function named(text: string): ActionLabel {
  return { text, shortcut: null };
}

/** The label read out in one line, as a screen reader hears it. */
export function spokenLabel(label: ActionLabel): string {
  return label.shortcut ? `${label.text} · ${label.shortcut}` : label.text;
}

/** A meeting button before the disclosure is acknowledged opens the Meetings
 *  page, where the first start shows it, rather than asking the backend for a
 *  refusal; a stop is always sent. The language button swaps back to the
 *  language used before, and opens Dictation when there is none. */
export function stepFor(action: RingAction, state: RingState): RingStep {
  switch (action) {
    case "meeting":
      if (state.meetingRecording || !state.settings || state.settings.meeting_consent_acknowledged) {
        return { kind: "toggle-meeting" };
      }
      return { kind: "open-panel", target: { panel: "meetings" } };
    case "file":
      return { kind: "pick-file" };
    case "dictate":
      return { kind: "toggle-dictation" };
    case "language": {
      const previous = state.settings?.previous_language;
      if (previous && previous !== state.settings?.language) {
        return { kind: "switch-language", language: previous };
      }
      return { kind: "open-panel", target: { panel: "dictation" } };
    }
    case "settings":
      return { kind: "open-panel", target: { panel: "settings" } };
  }
}

/** What the system notification says when `step` failed; the column has closed
 *  by then, so no window is left to say it in. */
export function failureText(step: RingStep, state: RingState): string {
  switch (step.kind) {
    case "toggle-meeting":
      return state.meetingRecording
        ? "Couldn't stop the meeting — try again."
        : "Couldn't start the meeting — try again.";
    case "pick-file":
      return "Couldn't open the file picker — try again.";
    case "toggle-dictation":
      return "Couldn't start dictating — try again.";
    case "switch-language":
      return `Couldn't switch to ${languageName(step.language)} — try again.`;
    case "open-panel":
      return "Couldn't open JustSay — try again.";
  }
}
