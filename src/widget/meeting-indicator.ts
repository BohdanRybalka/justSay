/**
 * The widget's meeting-recording indicator.
 *
 * ADR 040 obligation 2 requires a persistent, product-owned visible indicator
 * for the whole duration of a meeting recording, and ADR 041 records that the
 * macOS Core Audio tap produces no menu-bar indicator of its own — so this is
 * the only thing anyone can see. It lives here rather than inside widget.ts so
 * that "the indicator is showing" is assertable against a DOM without booting
 * the whole widget.
 *
 * It draws into its own `.pill-meeting` slot beside the pill's dictation slot,
 * never inside it: every dictation state change repaints that slot, so a
 * readout inside it would blank a running meeting timer until the next tick. A
 * marker that blinks is weaker evidence than one that does not, and this
 * marker is what the consent story rests on.
 */

import { formatElapsedClock } from "../format";
import { icon } from "../ui/icons";
import { scaledBars } from "./pill";

export const MEETING_STATE_CLASS = "meeting";

export const MEETING_SLOT_CLASS = "pill-meeting";

export const MEETING_STOP_SELECTOR = ".pill-stop";

/** Set on a meter whose side is no longer being captured, and on the mark when
 *  the recording itself is failing. A class rather than a different state,
 *  because the marker must not disappear or become something else: the call is
 *  still being recorded and the obligation still holds. */
export const MEETING_LOST_CLASS = "pill-lost";

export const MIC_BASE_HEIGHTS = [4, 8, 5] as const;

export const SYSTEM_BASE_HEIGHTS = [6, 4, 9] as const;

export type MeetingIncidentSide = "mic" | "system" | "recording";

export interface MeetingIndicatorState {
  active: boolean;
  elapsedSeconds: number;
  incident: string | null;
  /** 0..1, like the dictation wave's level. */
  mic: number;
  system: number;
}

const INCIDENT_SIDES: Record<string, MeetingIncidentSide> = {
  microphone_stalled: "mic",
  system_audio_ended: "system",
};

/** Which part of the pill an incident belongs to. Anything not about one side
 *  — the storage incidents, or a token this build does not know — is about the
 *  recording as a whole. */
export function incidentSide(incident: string): MeetingIncidentSide {
  return INCIDENT_SIDES[incident] ?? "recording";
}

function bars(count: number): string {
  return `<span class="pill-meter-bars">${"<i></i>".repeat(count)}</span>`;
}

const SKELETON =
  `<span class="pill-meeting-mark"><span class="pill-dot"></span>${icon("alert", "small")}</span>` +
  '<span class="pill-label pill-readout num"></span>' +
  '<span class="pill-divider"></span>' +
  `<span class="pill-meter" data-side="mic">${icon("mic", "small")}${icon("alert", "small")}` +
  `${bars(MIC_BASE_HEIGHTS.length)}</span>` +
  `<span class="pill-meter" data-side="system">${icon("speaker", "small")}${icon("alert", "small")}` +
  `${bars(SYSTEM_BASE_HEIGHTS.length)}</span>` +
  `<button class="pill-stop" type="button" aria-label="Stop recording the meeting">${icon("stop", "small")}</button>`;

function paintMeter(
  meter: HTMLElement,
  bases: readonly number[],
  level: number,
  lost: boolean,
): void {
  meter.classList.toggle(MEETING_LOST_CLASS, lost);
  const heights = scaledBars(bases, lost ? 0 : level);
  meter.querySelectorAll<HTMLElement>(".pill-meter-bars i").forEach((bar, index) => {
    bar.style.height = `${heights[index]}px`;
  });
}

export function renderMeetingIndicator(root: HTMLElement, state: MeetingIndicatorState): void {
  const slot = root.querySelector<HTMLElement>(`.${MEETING_SLOT_CLASS}`);
  root.classList.toggle(MEETING_STATE_CLASS, state.active);

  if (!slot) {
    if (state.active) {
      console.error(
        `The meeting recording indicator has no .${MEETING_SLOT_CLASS} to draw into, so the ` +
          "elapsed time ADR 040 obligation 2 requires is not being shown.",
      );
    }
    return;
  }
  if (!slot.firstChild) slot.innerHTML = SKELETON;

  const side = state.active && state.incident !== null ? incidentSide(state.incident) : null;
  slot.querySelector(".pill-meeting-mark")!.classList.toggle(MEETING_LOST_CLASS, side === "recording");
  slot.querySelector(".pill-readout")!.textContent = state.active
    ? formatElapsedClock(state.elapsedSeconds)
    : "";
  paintMeter(
    slot.querySelector<HTMLElement>('[data-side="mic"]')!,
    MIC_BASE_HEIGHTS,
    state.active ? state.mic : 0,
    side === "mic",
  );
  paintMeter(
    slot.querySelector<HTMLElement>('[data-side="system"]')!,
    SYSTEM_BASE_HEIGHTS,
    state.active ? state.system : 0,
    side === "system",
  );
}
