/**
 * The widget's pill, drawn from one view state. `root` is the `.pill` element
 * holding a `.pill-content` slot; only that slot and the pill's state
 * modifiers are written, so anything else inside the pill survives every
 * repaint — the meeting clock keeps its own readout there for that reason.
 */

import { icon } from "../ui/icons";

export type PillView =
  | { kind: "rest"; hint: string }
  | { kind: "listening" | "working" | "done"; label: string; readout: string }
  | { kind: "alert"; label: string };

/** Set by the shell's cursor loop while the pointer is near the pill; never
 *  touched by `renderPill`. */
export const PILL_HOVER_CLASS = "pill--hover";

const MODIFIER_BY_KIND: Record<PillView["kind"], string | null> = {
  rest: "pill--rest",
  listening: "pill--live",
  working: null,
  done: null,
  alert: "pill--alert",
};

const STATE_MODIFIERS = Object.values(MODIFIER_BY_KIND).filter(
  (modifier): modifier is string => modifier !== null,
);

function textPart(doc: Document, className: string, text: string): HTMLSpanElement {
  const span = doc.createElement("span");
  span.className = className;
  span.textContent = text;
  return span;
}

export function renderPill(root: HTMLElement, view: PillView): void {
  const modifier = MODIFIER_BY_KIND[view.kind];
  for (const each of STATE_MODIFIERS) root.classList.toggle(each, each === modifier);

  const slot = root.querySelector<HTMLElement>(":scope > .pill-content")!;
  const doc = root.ownerDocument;
  if (view.kind === "rest") {
    slot.innerHTML = icon("mic");
    slot.append(textPart(doc, "pill-label pill-hint", view.hint));
    return;
  }
  slot.replaceChildren(textPart(doc, "pill-label", view.label));
  if (view.kind !== "alert" && view.readout) {
    slot.append(textPart(doc, "pill-label pill-readout num", view.readout));
  }
}
