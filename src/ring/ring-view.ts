/**
 * The widget's actions as a slim column docked on the screen edge, the mic in
 * the middle where the pill sits: drawn once, opened and closed as the shell
 * says, and hovered from the pointer position the shell reports.
 */

import type { RingPointer } from "../contracts";
import { icon, type IconName } from "../ui/icons";
import { spokenLabel, type ActionLabel, type RingAction } from "./ring-actions";

interface RingButton {
  icon: IconName;
  action: RingAction;
}

export const RING_BUTTONS: readonly RingButton[] = [
  { icon: "users", action: "meeting" },
  { icon: "upload", action: "file" },
  { icon: "mic", action: "dictate" },
  { icon: "globe", action: "language" },
  { icon: "cog", action: "settings" },
];

const MIC_ROW = RING_BUTTONS.findIndex((button) => button.action === "dictate");

/** A 28px button and the 2px gap below it. */
const ROW_PITCH = 30;

const COLUMN_WIDTH = 37;
const COLUMN_HALF_HEIGHT = 77;

export const RING_OPEN_CLASS = "ring--open";
export const BUTTON_HOVER_CLASS = "ring-button--hover";
export const LABEL_SHOWN_CLASS = "ring-label--shown";

/** The button under the pointer, given in logical pixels from the column's
 *  right side on the widget's centre line. A row takes in half of each gap,
 *  so moving along the column never passes over nothing. */
export function buttonAt(pointer: RingPointer): number | null {
  if (pointer.x < -COLUMN_WIDTH || pointer.x > 0 || Math.abs(pointer.y) > COLUMN_HALF_HEIGHT) return null;
  const row = Math.round(pointer.y / ROW_PITCH) + MIC_ROW;
  return Math.min(Math.max(row, 0), RING_BUTTONS.length - 1);
}

function buttons(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(".ring-button")];
}

/** `onPick` gets a left click on any button; `onDismiss` a right click on
 *  the mic, which closes the column the way a right click on the pill opened it. */
export function mountRing(
  root: HTMLElement,
  onPick: (action: RingAction) => void,
  onDismiss: () => void,
): void {
  const column = root.querySelector<HTMLElement>(".ring-column")!;
  RING_BUTTONS.forEach((entry, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === MIC_ROW ? "ring-button ring-button--mic" : "ring-button";
    button.style.setProperty("--row", String(index));
    button.innerHTML = icon(entry.icon);
    button.addEventListener("click", () => onPick(entry.action));
    if (index === MIC_ROW) button.addEventListener("contextmenu", () => onDismiss());
    column.appendChild(button);
  });
}

const labels = new WeakMap<HTMLElement, ActionLabel>();

/** Names each button; the label beside the column reads the hovered one's name. */
export function renderLabels(root: HTMLElement, labelOf: (action: RingAction) => ActionLabel): void {
  buttons(root).forEach((button, index) => {
    const label = labelOf(RING_BUTTONS[index].action);
    labels.set(button, label);
    button.setAttribute("aria-label", spokenLabel(label));
  });
  const hovered = hoveredButton(root);
  if (hovered !== null) hoverButton(root, hovered);
}

/** Starts from rest before opening, so every opening slides in again. */
export function openRing(root: HTMLElement): void {
  closeRing(root);
  void root.offsetWidth;
  root.classList.add(RING_OPEN_CLASS);
  root.setAttribute("aria-hidden", "false");
}

export function closeRing(root: HTMLElement): void {
  root.classList.remove(RING_OPEN_CLASS);
  root.setAttribute("aria-hidden", "true");
  hoverButton(root, null);
}

function hoveredButton(root: HTMLElement): number | null {
  const index = buttons(root).findIndex((button) => button.classList.contains(BUTTON_HOVER_CLASS));
  return index === -1 ? null : index;
}

export function hoverButton(root: HTMLElement, hovered: number | null): void {
  buttons(root).forEach((button, index) => button.classList.toggle(BUTTON_HOVER_CLASS, index === hovered));
  const chip = root.querySelector<HTMLElement>(".ring-label")!;
  chip.classList.toggle(LABEL_SHOWN_CLASS, hovered !== null);
  if (hovered === null) return;
  chip.style.setProperty("--row", String(hovered));
  const label = labels.get(buttons(root)[hovered]);
  chip.replaceChildren(label?.text ?? "");
  if (label?.shortcut) {
    const shortcut = document.createElement("span");
    shortcut.className = "ring-label-shortcut";
    shortcut.textContent = ` · ${label.shortcut}`;
    chip.appendChild(shortcut);
  }
}

export function followPointer(root: HTMLElement, pointer: RingPointer): void {
  hoverButton(root, buttonAt(pointer));
}
