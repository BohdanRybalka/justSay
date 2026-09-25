/**
 * The widget's pill, drawn from one view state. `root` is the `.pill` element
 * holding a `.pill-content` slot; only that slot and the pill's state
 * modifiers are written, so anything else inside the pill survives every
 * repaint — the meeting clock keeps its own readout there for that reason.
 */

import { formatElapsedClock } from "../format";
import { icon } from "../ui/icons";

export type PillView =
  | { kind: "rest"; hint: string }
  | { kind: "listening"; elapsedSeconds: number; level: number }
  | { kind: "working" }
  | { kind: "done"; words: number }
  | { kind: "noSpeech" }
  | { kind: "alert"; label: string };

/** Set by the shell's cursor loop while the pointer is near the pill; never
 *  touched by `renderPill`. */
export const PILL_HOVER_CLASS = "pill--hover";

export const WAVE_BASE_HEIGHTS = [4, 8, 11, 6, 10, 5, 7] as const;

const WAVE_FLOOR_PX = 2;

const MODIFIER_BY_KIND: Record<PillView["kind"], string | null> = {
  rest: "pill--rest",
  listening: "pill--live",
  working: null,
  done: null,
  noSpeech: null,
  alert: "pill--alert",
};

const STATE_MODIFIERS = Object.values(MODIFIER_BY_KIND).filter(
  (modifier): modifier is string => modifier !== null,
);

const SKELETON: Record<PillView["kind"], string> = {
  rest: `${icon("mic")}<span class="pill-label pill-hint"></span>`,
  listening:
    '<span class="pill-dot"></span>' +
    `<span class="pill-wave">${"<i></i>".repeat(WAVE_BASE_HEIGHTS.length)}</span>` +
    '<span class="pill-label pill-readout num"></span>',
  working:
    '<span class="pill-dots"><i></i><i></i><i></i></span>' +
    '<span class="pill-label pill-note">Writing it down</span>',
  done:
    `<span class="pill-mark pill-mark--done">${icon("check", "small")}</span>` +
    '<span class="pill-label"><span class="num pill-count"></span><span class="pill-unit"></span></span>',
  noSpeech: '<span class="pill-label pill-note">No speech</span>',
  alert:
    `<span class="pill-mark pill-mark--alert">${icon("alert", "small")}</span>` +
    '<span class="pill-label"></span>',
};

/** Bar heights in px for a level in 0..1: flat at silence, the design's
 *  `bases` at full voice, and the same bars either way so the pill's width
 *  never moves. */
export function scaledBars(bases: readonly number[], level: number): number[] {
  const clamped = Math.min(1, Math.max(0, level));
  return bases.map((base) => WAVE_FLOOR_PX + (base - WAVE_FLOOR_PX) * clamped);
}

export function waveHeights(level: number): number[] {
  return scaledBars(WAVE_BASE_HEIGHTS, level);
}

function part(slot: HTMLElement, selector: string): HTMLElement {
  return slot.querySelector<HTMLElement>(selector)!;
}

function fill(slot: HTMLElement, view: PillView): void {
  switch (view.kind) {
    case "rest":
      part(slot, ".pill-hint").textContent = view.hint;
      return;
    case "listening": {
      const heights = waveHeights(view.level);
      slot.querySelectorAll<HTMLElement>(".pill-wave i").forEach((bar, index) => {
        bar.style.height = `${heights[index]}px`;
      });
      part(slot, ".pill-readout").textContent = formatElapsedClock(view.elapsedSeconds);
      return;
    }
    case "done":
      part(slot, ".pill-count").textContent = String(view.words);
      part(slot, ".pill-unit").textContent = view.words === 1 ? " word · copied" : " words · copied";
      return;
    case "alert":
      part(slot, ".pill-label").textContent = view.label;
      return;
  }
}

/** A repaint of the same kind only refills its text and bar heights, so the
 *  blinking dot and the pulsing dots keep their animation through the ten
 *  level updates a second that listening receives. */
export function renderPill(root: HTMLElement, view: PillView): void {
  const modifier = MODIFIER_BY_KIND[view.kind];
  for (const each of STATE_MODIFIERS) root.classList.toggle(each, each === modifier);

  const slot = root.querySelector<HTMLElement>(":scope > .pill-content")!;
  if (slot.dataset.view !== view.kind) {
    slot.innerHTML = SKELETON[view.kind];
    slot.dataset.view = view.kind;
  }
  fill(slot, view);
}
