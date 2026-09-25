// @vitest-environment jsdom
import { readFileSync } from "fs";
import { join } from "path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderFold, renderSegmented, renderSelect, renderToggle } from "./controls";

type Range = "7d" | "30d" | "90d";
const RANGES = [
  { value: "7d", label: "7 days" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
] as const;

function mountSegmented(value: Range, onChange = vi.fn()): { root: HTMLElement; buttons: HTMLButtonElement[] } {
  const root = document.createElement("div");
  document.body.append(root);
  renderSegmented<Range>(root, RANGES, value, onChange);
  return { root, buttons: [...root.querySelectorAll("button")] };
}

function pressedLabels(root: HTMLElement): string[] {
  return [...root.querySelectorAll('button[aria-pressed="true"]')].map((b) => b.textContent ?? "");
}

function pressKey(target: HTMLElement, key: string): KeyboardEvent {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  return event;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("renderSegmented", () => {
  it("draws one button per option with only the current value pressed", () => {
    const { root, buttons } = mountSegmented("30d");

    expect(root.classList.contains("segmented")).toBe(true);
    expect(root.getAttribute("role")).toBe("group");
    expect(buttons.map((b) => b.textContent)).toEqual(["7 days", "30 days", "90 days"]);
    expect(buttons.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true", "false"]);
  });

  it("keeps only the pressed button in the tab order", () => {
    const { buttons } = mountSegmented("30d");

    expect(buttons.map((b) => b.tabIndex)).toEqual([-1, 0, -1]);
  });

  it("presses a clicked option and reports its value", () => {
    const onChange = vi.fn();
    const { root, buttons } = mountSegmented("30d", onChange);

    buttons[0].click();

    expect(pressedLabels(root)).toEqual(["7 days"]);
    expect(buttons.map((b) => b.tabIndex)).toEqual([0, -1, -1]);
    expect(onChange).toHaveBeenCalledExactlyOnceWith("7d");
  });

  it("stays quiet when the pressed option is clicked again", () => {
    const onChange = vi.fn();
    const { buttons } = mountSegmented("30d", onChange);

    buttons[1].click();

    expect(onChange).not.toHaveBeenCalled();
  });

  it("moves the choice and the focus with the arrow keys, wrapping at the ends", () => {
    const onChange = vi.fn();
    const { root, buttons } = mountSegmented("90d", onChange);

    const right = pressKey(buttons[2], "ArrowRight");
    expect(right.defaultPrevented).toBe(true);
    expect(pressedLabels(root)).toEqual(["7 days"]);
    expect(document.activeElement).toBe(buttons[0]);

    pressKey(buttons[0], "ArrowLeft");
    expect(pressedLabels(root)).toEqual(["90 days"]);
    expect(document.activeElement).toBe(buttons[2]);
    expect(onChange.mock.calls).toEqual([["7d"], ["90d"]]);
  });

  it("keeps the focus on the chosen option when the change redraws the control", () => {
    const root = document.createElement("div");
    document.body.append(root);
    const redraw = (value: Range): void => renderSegmented<Range>(root, RANGES, value, redraw);
    redraw("7d");

    pressKey(root.querySelectorAll("button")[0], "ArrowRight");

    expect(pressedLabels(root)).toEqual(["30 days"]);
    expect(document.activeElement).toBe(root.querySelector('button[aria-pressed="true"]'));
  });

  it("ignores keys other than the arrows", () => {
    const onChange = vi.fn();
    const { root, buttons } = mountSegmented("30d", onChange);

    expect(pressKey(buttons[1], "Tab").defaultPrevented).toBe(false);
    expect(pressedLabels(root)).toEqual(["30 days"]);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("redraws instead of adding buttons when called again", () => {
    const { root } = mountSegmented("30d");

    renderSegmented<Range>(root, RANGES, "7d", vi.fn());

    expect(root.querySelectorAll("button")).toHaveLength(3);
    expect(pressedLabels(root)).toEqual(["7 days"]);
  });

  it("leaves the first button reachable by Tab when the value matches no option", () => {
    const { root, buttons } = mountSegmented("1y" as Range);

    expect(pressedLabels(root)).toEqual([]);
    expect(buttons.map((b) => b.tabIndex)).toEqual([0, -1, -1]);
  });
});

describe("renderToggle", () => {
  function mountToggle(on: boolean, onChange = vi.fn()): HTMLButtonElement {
    const button = document.createElement("button");
    document.body.append(button);
    renderToggle(button, on, onChange);
    return button;
  }

  it("is a switch showing the given state", () => {
    const button = mountToggle(true);

    expect(button.classList.contains("toggle")).toBe(true);
    expect(button.type).toBe("button");
    expect(button.getAttribute("role")).toBe("switch");
    expect(button.getAttribute("aria-checked")).toBe("true");
  });

  it("flips on each click and reports the new state", () => {
    const onChange = vi.fn();
    const button = mountToggle(false, onChange);

    button.click();
    expect(button.getAttribute("aria-checked")).toBe("true");
    button.click();
    expect(button.getAttribute("aria-checked")).toBe("false");

    expect(onChange.mock.calls).toEqual([[true], [false]]);
  });

  it("reports to the latest handler only when rendered again", () => {
    const first = vi.fn();
    const second = vi.fn();
    const button = mountToggle(false, first);

    renderToggle(button, true, second);
    button.click();

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledExactlyOnceWith(false);
  });
});

describe("renderFold", () => {
  function mountDetails(): HTMLDetailsElement {
    document.body.innerHTML = "<details><summary>API keys</summary><p>Groq</p></details>";
    return document.querySelector("details")!;
  }

  it("puts one chevron before the summary's text", () => {
    const details = mountDetails();

    renderFold(details);
    renderFold(details);

    const summary = details.querySelector("summary")!;
    expect(details.classList.contains("fold")).toBe(true);
    expect(summary.querySelectorAll("svg.icon")).toHaveLength(1);
    expect(summary.firstElementChild?.querySelector("use")?.getAttribute("href")).toBe("#chev");
    expect(summary.textContent).toBe("API keys");
  });
});

describe("renderSelect", () => {
  function mountSelect(): HTMLSelectElement {
    document.body.innerHTML =
      '<div id="row"><select><option value="en">English</option><option value="uk" selected>Ukrainian</option></select></div>';
    return document.querySelector("select")!;
  }

  it("wraps the native select in the select box with a chevron after it", () => {
    const select = mountSelect();

    renderSelect(select);

    const box = select.parentElement!;
    expect(box.tagName).toBe("SPAN");
    expect(box.classList.contains("select")).toBe(true);
    expect(box.parentElement?.id).toBe("row");
    expect(select.nextElementSibling?.querySelector("use")?.getAttribute("href")).toBe("#chev");
    expect(select.value).toBe("uk");
  });

  it("wraps only once when called again", () => {
    const select = mountSelect();

    renderSelect(select);
    renderSelect(select);

    expect(document.querySelectorAll(".select")).toHaveLength(1);
    expect(document.querySelectorAll(".select svg")).toHaveLength(1);
  });
});

describe("controls stylesheet", () => {
  const css = readFileSync(join(__dirname, "controls.css"), "utf-8");

  it.each([
    '.segmented button[aria-pressed="true"]',
    '.toggle[aria-checked="true"]',
    ".fold[open]>summary .icon",
    ".select select",
    ".select .icon",
  ])("styles the state the helpers set: %s", (selector) => {
    expect(css).toContain(`${selector}{`);
  });
});
