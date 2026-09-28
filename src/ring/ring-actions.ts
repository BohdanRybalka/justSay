/**
 * What each petal says and what picking it does, decided from the settings and
 * the meeting state the ring last read. Nothing here talks to a window or the
 * backend; `ring.ts` carries out the step this module names.
 */

import { meetingsTurnedOn, type UserSettings } from "../api";
import type { NavigatePanel } from "../contracts";
import { languageName } from "../languages";

export type RingAction = "meeting" | "file" | "language" | "settings";

export interface RingState {
  settings: Pick<
    UserSettings,
    "language" | "previous_language" | "meeting_consent_acknowledged" | "meetings_enabled"
  > | null;
  meetingRecording: boolean;
}

export type RingStep =
  | { kind: "toggle-meeting" }
  | { kind: "switch-language"; language: string }
  | { kind: "open-panel"; target: NavigatePanel };

export const UNREAD_RING_STATE: RingState = { settings: null, meetingRecording: false };

export function petalLabel(action: RingAction, state: RingState): string {
  switch (action) {
    case "meeting":
      return state.meetingRecording ? "Stop the meeting" : "Record a meeting";
    case "file":
      return "Transcribe a file";
    case "language":
      return state.settings ? `Language · ${languageName(state.settings.language)}` : "Language";
    case "settings":
      return "Settings";
  }
}

/** A meeting petal with meetings off opens their switch rather than asking the
 *  backend for a refusal; a stop is always sent. The language petal swaps back
 *  to the language used before, and opens Dictation when there is none. */
export function stepFor(action: RingAction, state: RingState): RingStep {
  switch (action) {
    case "meeting":
      if (state.meetingRecording || !state.settings || meetingsTurnedOn(state.settings)) {
        return { kind: "toggle-meeting" };
      }
      return { kind: "open-panel", target: { panel: "dictation", section: "meetings" } };
    case "file":
      return { kind: "open-panel", target: { panel: "history", section: null } };
    case "language": {
      const previous = state.settings?.previous_language;
      if (previous && previous !== state.settings?.language) {
        return { kind: "switch-language", language: previous };
      }
      return { kind: "open-panel", target: { panel: "dictation", section: null } };
    }
    case "settings":
      return { kind: "open-panel", target: { panel: "settings", section: null } };
  }
}
