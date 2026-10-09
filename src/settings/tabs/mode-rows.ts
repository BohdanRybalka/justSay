/** The big radio rows that pick Cloud or Local model, and what each says about Cloud. */
import type { CloudKeyStatus } from "../../api";
import { icon } from "../../ui/icons";

export const CLOUD_KEY_MISSING = "Add a Groq key in Settings";

/** Whether Cloud's only engine, Groq, has no key; an unread key status is not missing. */
export function cloudKeyMissing(cloud: CloudKeyStatus | null): boolean {
  return cloud !== null && !cloud.groq_key_set;
}

export interface RowView {
  hint: string;
  alert: boolean;
  locked: boolean;
  disabled: boolean;
  action: string;
}

export function drawRow(row: HTMLButtonElement, checked: boolean, view: RowView): void {
  row.setAttribute("aria-checked", String(checked));
  row.classList.toggle("mode-row--locked", view.locked);
  if (view.disabled) row.setAttribute("aria-disabled", "true");
  else row.removeAttribute("aria-disabled");
  const hint = row.querySelector<HTMLElement>(".mode-row-hint")!;
  hint.textContent = view.hint;
  hint.classList.toggle("mode-row-alert", view.alert);
  row.querySelector<HTMLElement>(".mode-row-action")!.innerHTML = view.action;
}

export function modeRowHtml(id: string, iconName: "cloud" | "chip", title: string): string {
  return `
    <button type="button" class="mode-row" id="${id}" role="radio" aria-checked="false">
      <span class="mode-row-radio"><i></i></span>
      <span class="mode-row-icon">${icon(iconName, "large")}</span>
      <span class="mode-row-text"><b>${title}</b><small class="mode-row-hint"></small></span>
      <span class="setting-row-controls mode-row-action"></span>
    </button>`;
}
