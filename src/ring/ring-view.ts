/**
 * The ring's four petals around its centre: drawn once, opened and closed as
 * the shell says, and hovered from the pointer position the shell reports.
 */

import type { RingPointer } from "../contracts";
import { icon, type IconName } from "../ui/icons";

interface RingPetal {
  icon: IconName;
  label: string;
}

export const RING_PETALS: readonly RingPetal[] = [
  { icon: "users", label: "Record a meeting" },
  { icon: "upload", label: "Transcribe a file" },
  { icon: "globe", label: "Language · English" },
  { icon: "cog", label: "Settings" },
];

const RING_RADIUS = 38;
const PETAL_RADIUS = 14;
const HOVERED_PETAL_SCALE = 1.24;

export const RING_OPEN_CLASS = "ring--open";
export const PETAL_HOVER_CLASS = "ring-petal--hover";
export const LABEL_SHOWN_CLASS = "ring-label--shown";

/** Degrees clockwise from the right; the first petal sits straight above. */
function petalAngle(index: number): number {
  return (index * 360) / RING_PETALS.length - 90;
}

export function petalCentre(index: number): RingPointer {
  const radians = (petalAngle(index) * Math.PI) / 180;
  return { x: RING_RADIUS * Math.cos(radians), y: RING_RADIUS * Math.sin(radians) };
}

/** The petal under the pointer, measured as drawn: the hovered one is grown,
 *  so the pointer leaves it where its enlarged edge is, not its resting one. */
export function petalAt(pointer: RingPointer, hovered: number | null): number | null {
  const index = RING_PETALS.findIndex((_, i) => {
    const centre = petalCentre(i);
    const reach = PETAL_RADIUS * (i === hovered ? HOVERED_PETAL_SCALE : 1);
    return Math.hypot(pointer.x - centre.x, pointer.y - centre.y) <= reach;
  });
  return index === -1 ? null : index;
}

function petals(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(".ring-petal")];
}

export function mountRing(root: HTMLElement, onPick: (index: number) => void): void {
  RING_PETALS.forEach((petal, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ring-petal";
    button.style.setProperty("--i", String(index));
    button.style.setProperty("--a", `${petalAngle(index)}deg`);
    button.setAttribute("aria-label", petal.label);
    button.innerHTML = icon(petal.icon);
    button.addEventListener("click", () => onPick(index));
    root.appendChild(button);
  });
}

/** Starts from rest before opening, so every opening fans out again. */
export function openRing(root: HTMLElement): void {
  closeRing(root);
  void root.offsetWidth;
  root.classList.add(RING_OPEN_CLASS);
  root.setAttribute("aria-hidden", "false");
}

export function closeRing(root: HTMLElement): void {
  root.classList.remove(RING_OPEN_CLASS);
  root.setAttribute("aria-hidden", "true");
  hoverPetal(root, null);
}

export function hoverPetal(root: HTMLElement, hovered: number | null): void {
  petals(root).forEach((petal, index) => petal.classList.toggle(PETAL_HOVER_CLASS, index === hovered));
  const label = root.querySelector<HTMLElement>(".ring-label")!;
  label.classList.toggle(LABEL_SHOWN_CLASS, hovered !== null);
  if (hovered !== null) label.textContent = RING_PETALS[hovered].label;
}

export function followPointer(root: HTMLElement, pointer: RingPointer): void {
  const hovered = petals(root).findIndex((petal) => petal.classList.contains(PETAL_HOVER_CLASS));
  hoverPetal(root, petalAt(pointer, hovered === -1 ? null : hovered));
}
