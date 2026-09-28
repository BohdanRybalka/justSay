// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  closeRing,
  followPointer,
  LABEL_SHOWN_CLASS,
  mountRing,
  openRing,
  PETAL_HOVER_CLASS,
  petalAt,
  petalCentre,
  RING_OPEN_CLASS,
  RING_PETALS,
} from "./ring-view";

function ringRoot(): HTMLElement {
  const html = readFileSync(resolve(__dirname, "../../ring.html"), "utf-8");
  document.body.innerHTML = html
    .slice(html.indexOf("<body>") + "<body>".length, html.indexOf("</body>"))
    .replace(/<script[\s\S]*?<\/script>/g, "");
  return document.getElementById("ring")!;
}

function hoveredPetals(root: HTMLElement): number[] {
  return [...root.querySelectorAll(".ring-petal")].flatMap((petal, index) =>
    petal.classList.contains(PETAL_HOVER_CLASS) ? [index] : [],
  );
}

function label(root: HTMLElement): { text: string | null; shown: boolean } {
  const element = root.querySelector(".ring-label")!;
  return { text: element.textContent, shown: element.classList.contains(LABEL_SHOWN_CLASS) };
}

describe("the ring's geometry", () => {
  it("puts five petals 38px out, 72° apart, the first straight above the centre", () => {
    expect(RING_PETALS).toHaveLength(5);
    const [top, second] = [petalCentre(0), petalCentre(1)];
    expect(top.x).toBeCloseTo(0);
    expect(top.y).toBeCloseTo(-38);
    expect(Math.atan2(second.y, second.x) * (180 / Math.PI)).toBeCloseTo(-18);
    for (let i = 0; i < 5; i++) {
      expect(Math.hypot(petalCentre(i).x, petalCentre(i).y)).toBeCloseTo(38);
    }
  });

  it("finds the petal under the pointer and none between them or at the centre", () => {
    expect(petalAt({ x: 0, y: -38 }, null)).toBe(0);
    expect(petalAt({ x: 0, y: -52 }, null)).toBe(0);
    expect(petalAt({ x: 0, y: 0 }, null)).toBeNull();
    const betweenFirstTwo = { x: 38 * Math.cos(-0.94), y: 38 * Math.sin(-0.94) };
    expect(petalAt(betweenFirstTwo, null)).toBeNull();
  });

  it("keeps a grown petal hovered out to its enlarged edge", () => {
    const pastRestingEdge = { x: 0, y: -38 - 16 };

    expect(petalAt(pastRestingEdge, null)).toBeNull();
    expect(petalAt(pastRestingEdge, 0)).toBe(0);
    expect(petalAt(pastRestingEdge, 3)).toBeNull();
  });
});

describe("the ring on the page", () => {
  let root: HTMLElement;
  const onPick = vi.fn();

  beforeEach(() => {
    onPick.mockClear();
    root = ringRoot();
    mountRing(root, onPick);
  });

  it("draws one labelled button per action, in order, around the centre", () => {
    const petals = [...root.querySelectorAll<HTMLElement>(".ring-petal")];

    expect(petals.map((petal) => petal.getAttribute("aria-label"))).toEqual([
      "Record a meeting",
      "Transcribe a file",
      "Language · English",
      "Settings",
      "Processed in the cloud",
    ]);
    expect(petals.map((petal) => petal.style.getPropertyValue("--a"))).toEqual([
      "-90deg",
      "-18deg",
      "54deg",
      "126deg",
      "198deg",
    ]);
    expect(petals.map((petal) => petal.style.getPropertyValue("--i"))).toEqual(["0", "1", "2", "3", "4"]);
  });

  it("grows the petal under the pointer and names it in the label", () => {
    openRing(root);

    followPointer(root, { x: 0, y: -40 });
    expect(hoveredPetals(root)).toEqual([0]);
    expect(label(root)).toEqual({ text: "Record a meeting", shown: true });

    followPointer(root, petalCentre(3));
    expect(hoveredPetals(root)).toEqual([3]);
    expect(label(root)).toEqual({ text: "Settings", shown: true });

    followPointer(root, { x: 0, y: 0 });
    expect(hoveredPetals(root)).toEqual([]);
    expect(label(root).shown).toBe(false);
  });

  it("hands the picked petal to its caller", () => {
    openRing(root);

    root.querySelectorAll<HTMLElement>(".ring-petal")[2].click();

    expect(onPick).toHaveBeenCalledWith(2);
  });

  it("closes back to rest, so the next opening fans out with nothing hovered", () => {
    openRing(root);
    followPointer(root, { x: 0, y: -38 });

    closeRing(root);

    expect(root.classList.contains(RING_OPEN_CLASS)).toBe(false);
    expect(root.getAttribute("aria-hidden")).toBe("true");
    expect(hoveredPetals(root)).toEqual([]);
    expect(label(root).shown).toBe(false);

    openRing(root);
    expect(root.classList.contains(RING_OPEN_CLASS)).toBe(true);
    expect(root.getAttribute("aria-hidden")).toBe("false");
    expect(hoveredPetals(root)).toEqual([]);
  });

  it("starts an opening from rest even when the ring was never told it closed", () => {
    openRing(root);
    followPointer(root, { x: 0, y: -38 });

    openRing(root);

    expect(hoveredPetals(root)).toEqual([]);
  });
});
