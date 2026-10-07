/**
 * The API keys fold: one row per cloud key. A row is stored, env, unset,
 * unknown or editing, and is redrawn alone; its subtitle says what the key is
 * used for, and in Local mode that it is not used.
 */
import { type CloudKeyStatus, type UserSettings } from "../../api";
import { MASKED_API_KEY } from "../../contracts";
import { saveSettings, getCloudKeyStatus } from "../settings";

type KeyField = "gemini_api_key" | "groq_api_key";
type KeyRowState = "stored" | "env" | "unset" | "unknown" | "editing";

interface KeyRowSpec {
  field: KeyField;
  provider: "groq" | "gemini";
  label: string;
  use: string;
}

const ROWS: readonly KeyRowSpec[] = [
  { field: "groq_api_key", provider: "groq", label: "Groq", use: "Turns your recordings into text" },
  { field: "gemini_api_key", provider: "gemini", label: "Google", use: "Used for History search" },
];

const LOCAL_HINT = "Not used while Local is on";

const STATE_HINTS: Readonly<Partial<Record<KeyRowState, string>>> = {
  env: "Key active (from environment). Saving a key here will override it.",
  unknown: "Cannot verify key status — reopen Settings to retry.",
};

const MASKED_DISPLAY = "••••••••••••";

function describeFailure(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function cloudFlag(cloud: CloudKeyStatus, field: KeyField): boolean {
  return field === "gemini_api_key" ? cloud.gemini_key_set : cloud.groq_key_set;
}

function rowState(settings: UserSettings, field: KeyField, cloud: CloudKeyStatus | null): KeyRowState {
  if (settings[field] === MASKED_API_KEY) return "stored";
  if (cloud === null) return "unknown";
  return cloudFlag(cloud, field) ? "env" : "unset";
}

function useHint(settings: UserSettings, spec: KeyRowSpec): string {
  return settings.stt_mode === "local" ? LOCAL_HINT : spec.use;
}

function keyControls(spec: KeyRowSpec, state: KeyRowState): string {
  const p = spec.provider;
  if (state === "stored" || state === "env") {
    return `
      <input class="masked-field num" id="${p}-key-input" value="${MASKED_DISPLAY}" readonly
        aria-label="${spec.label} key, hidden" />
      <button type="button" class="btn" id="${p}-replace">Replace</button>
    `;
  }
  const cancel = state === "editing" ? `<button type="button" class="btn" id="${p}-cancel">Cancel</button>` : "";
  return `
    <input class="masked-field" type="password" id="${p}-key-input" autocomplete="off" spellcheck="false"
      placeholder="Paste your ${spec.label} key" aria-label="${spec.label} key" />
    ${cancel}
    <button type="button" class="btn btn-primary" id="${p}-save" disabled>Save</button>
  `;
}

/** Fills `container` with the Groq and Google rows; Save goes through `saveSettings`. */
export function renderKeys(container: HTMLElement, settings: UserSettings, cloud: CloudKeyStatus | null): void {
  let current = settings;
  let knownCloud = cloud;
  container.innerHTML = `
    ${ROWS.map((spec) => `<div class="setting-row key-row" data-provider="${spec.provider}"></div>`).join("")}
  `;
  const rowOf = (spec: KeyRowSpec): HTMLElement =>
    container.querySelector<HTMLElement>(`.key-row[data-provider="${spec.provider}"]`)!;

  const draw = (spec: KeyRowSpec, state: KeyRowState, refocus?: boolean): void => {
    const row = rowOf(spec);
    const hadFocus = refocus ?? row.contains(document.activeElement);
    row.innerHTML = `
      <div class="setting-row-text">
        <div class="setting-row-title">${spec.label}</div>
        <div class="setting-row-hint use-hint">${useHint(current, spec)}</div>
        <div class="setting-row-hint key-status" id="${spec.provider}-status" aria-live="polite">${STATE_HINTS[state] ?? ""}</div>
      </div>
      <div class="setting-row-controls">${keyControls(spec, state)}</div>
    `;
    wireKey(spec, state, row);
    if (state === "editing") row.querySelector<HTMLInputElement>("input")!.focus();
    else if (hadFocus) row.querySelector<HTMLElement>("button")?.focus();
  };

  const wireKey = (spec: KeyRowSpec, state: KeyRowState, row: HTMLElement): void => {
    const p = spec.provider;
    const storedState = (): KeyRowState => rowState(current, spec.field, knownCloud);
    if (state === "stored" || state === "env") {
      row.querySelector(`#${p}-replace`)!.addEventListener("click", () => draw(spec, "editing"));
      return;
    }
    row.querySelector(`#${p}-cancel`)?.addEventListener("click", () => draw(spec, storedState()));

    const input = row.querySelector<HTMLInputElement>(`#${p}-key-input`)!;
    const save = row.querySelector<HTMLButtonElement>(`#${p}-save`)!;
    const status = row.querySelector<HTMLElement>(`#${p}-status`)!;
    const locked = [input, save, ...row.querySelectorAll<HTMLButtonElement>(`#${p}-cancel`)];
    input.addEventListener("input", () => {
      save.disabled = input.value.trim() === "";
    });
    save.addEventListener("click", async () => {
      const value = input.value.trim();
      if (!value) return;
      const refocus = row.contains(document.activeElement);
      locked.forEach((control) => (control.disabled = true));
      save.textContent = "Saving…";
      status.textContent = "";
      try {
        const { settings: fresh } = await saveSettings({ [spec.field]: value } as Partial<UserSettings>);
        current = fresh;
        knownCloud = getCloudKeyStatus();
        draw(spec, storedState(), refocus);
      } catch (err) {
        status.textContent = `Error: ${describeFailure(err)}`;
        locked.forEach((control) => (control.disabled = false));
        save.textContent = "Save";
      }
    });
  };

  for (const spec of ROWS) draw(spec, rowState(current, spec.field, knownCloud));
}
