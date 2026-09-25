/**
 * Behaviour for the design's shared controls, whose look lives in
 * `controls.css`. Each helper builds or updates the markup that stylesheet
 * expects and keeps its ARIA state; the caller owns the value and saves it.
 * Everything else about a control is plain markup with the stylesheet's classes.
 */

import { icon } from "./icons";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

const ARROW_STEP: Readonly<Record<string, number>> = { ArrowLeft: -1, ArrowRight: 1 };

/**
 * Fills `root` with one pressed-state button per option, `value` pressed.
 * A click or an arrow key presses another and calls `onChange`; only the
 * pressed button is in the tab order. Calling it again redraws the buttons.
 */
export function renderSegmented<T extends string>(
  root: HTMLElement,
  options: readonly SegmentedOption<T>[],
  value: T,
  onChange: (value: T) => void,
): void {
  const buttons = options.map(({ label }) => {
    const button = root.ownerDocument.createElement("button");
    button.type = "button";
    button.textContent = label;
    return button;
  });
  let pressed = options.findIndex((option) => option.value === value);
  const press = (index: number): void => {
    pressed = index;
    buttons.forEach((button, i) => {
      button.setAttribute("aria-pressed", String(i === index));
      button.tabIndex = i === Math.max(index, 0) ? 0 : -1;
    });
  };
  const choose = (index: number): void => {
    if (index === pressed) return;
    press(index);
    onChange(options[index].value);
  };
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => choose(index));
    button.addEventListener("keydown", (event) => {
      const step = ARROW_STEP[event.key];
      if (step === undefined) return;
      event.preventDefault();
      const next = (index + step + buttons.length) % buttons.length;
      choose(next);
      buttons[next].focus();
    });
  });
  root.classList.add("segmented");
  root.setAttribute("role", "group");
  root.replaceChildren(...buttons);
  press(pressed);
}

/**
 * Makes `button` an on/off switch showing `on`. A click, or Space and Enter
 * through the native button, flips it and calls `onChange` with the new
 * state. Calling it again replaces the previous `onChange`.
 */
export function renderToggle(
  button: HTMLButtonElement,
  on: boolean,
  onChange: (on: boolean) => void,
): void {
  button.type = "button";
  button.classList.add("toggle");
  button.setAttribute("role", "switch");
  button.setAttribute("aria-checked", String(on));
  button.onclick = () => {
    const next = button.getAttribute("aria-checked") !== "true";
    button.setAttribute("aria-checked", String(next));
    onChange(next);
  };
}

/** Styles a `<details>` as a collapsible section whose chevron turns when it opens. */
export function renderFold(details: HTMLDetailsElement): void {
  details.classList.add("fold");
  const summary = details.querySelector(":scope > summary");
  if (summary && !summary.querySelector(".icon")) {
    summary.insertAdjacentHTML("afterbegin", icon("chev", "small"));
  }
}

/** Wraps a native `<select>` that is in the document in the design's select box with its chevron. */
export function renderSelect(select: HTMLSelectElement): void {
  if (select.parentElement?.classList.contains("select")) return;
  const box = select.ownerDocument.createElement("span");
  box.className = "select";
  select.replaceWith(box);
  box.append(select);
  box.insertAdjacentHTML("beforeend", icon("chev", "small"));
}
