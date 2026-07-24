import { type CloudKeyStatus, type UserSettings } from "../../api";
import { saveSettings, getCloudKeyStatus } from "../settings";

type KeyField = "gemini_api_key" | "groq_api_key";
type KeyRowState = "stored" | "env" | "unset" | "unknown" | "editing";

interface KeyRowSpec {
  field: KeyField;
  label: string;
  unsetHint: string;
  storedHint: string;
  envHint: string;
  unknownHint: string;
  placeholder: string;
}

const ROWS: KeyRowSpec[] = [
  {
    field: "gemini_api_key",
    label: "Gemini (STT — long audio &amp; structured)",
    unsetHint: "No key set — cloud STT will fail.",
    storedHint: "Key stored.",
    envHint: "Key active (from environment). Saving a key here will override it.",
    unknownHint: "Cannot verify key status — reopen Settings to retry.",
    placeholder: "Paste your Gemini API key",
  },
  {
    field: "groq_api_key",
    label: "Groq (STT — short audio &amp; LLM)",
    unsetHint: "No key set — cloud STT and LLM will fail.",
    storedHint: "Key stored.",
    envHint: "Key active (from environment). Saving a key here will override it.",
    unknownHint: "Cannot verify key status — reopen Settings to retry.",
    placeholder: "Paste your Groq API key",
  },
];

const MASKED_DISPLAY = "••••••••";

function prefix(field: KeyField): string {
  return field === "gemini_api_key" ? "gemini" : "groq";
}

function cloudFlag(cloud: CloudKeyStatus, field: KeyField): boolean {
  return field === "gemini_api_key" ? cloud.gemini_key_set : cloud.groq_key_set;
}

function rowState(settings: UserSettings, field: KeyField, cloud: CloudKeyStatus | null): KeyRowState {
  if (settings[field] === "***") return "stored";
  if (cloud === null) return "unknown";
  return cloudFlag(cloud, field) ? "env" : "unset";
}

function renderRowMarkup(spec: KeyRowSpec, state: KeyRowState): string {
  const p = prefix(spec.field);
  let inner: string;
  if (state === "stored" || state === "env") {
    inner = `
      <input type="text" id="${p}-key-input" disabled
        value="${MASKED_DISPLAY}"
        aria-label="Stored API key (masked)" />
      <div class="key-actions">
        <button class="btn btn-secondary btn-sm" id="${p}-replace">Replace</button>
      </div>
    `;
  } else if (state === "editing") {
    inner = `
      <input type="password" id="${p}-key-input" autocomplete="off" spellcheck="false"
        placeholder="${spec.placeholder}" />
      <div class="key-actions">
        <button class="btn btn-secondary btn-sm" id="${p}-cancel">Cancel</button>
        <button class="btn btn-primary btn-sm" id="${p}-save" disabled>Save</button>
      </div>
    `;
  } else {
    inner = `
      <input type="password" id="${p}-key-input" autocomplete="off" spellcheck="false"
        placeholder="${spec.placeholder}" />
      <div class="key-actions">
        <button class="btn btn-primary btn-sm" id="${p}-save" disabled>Save</button>
      </div>
    `;
  }

  const statusText =
    state === "stored"
      ? spec.storedHint
      : state === "env"
        ? spec.envHint
        : state === "unknown"
          ? spec.unknownHint
          : state === "editing"
            ? "Editing stored key — Cancel to abort."
            : spec.unsetHint;

  return `
    <div class="setting-group">
      <div class="setting-label">${spec.label}</div>
      <div class="setting-row" style="flex-direction: column; align-items: stretch; gap: 8px;">
        <div class="key-input-row">
          ${inner}
        </div>
        <div class="setting-hint" id="${p}-status" aria-live="polite">${statusText}</div>
      </div>
    </div>
  `;
}

export function renderKeys(
  container: HTMLElement,
  settings: UserSettings,
  cloud: CloudKeyStatus | null,
  overrideStates?: Partial<Record<KeyField, KeyRowState>>,
): () => void {
  const states: Record<KeyField, KeyRowState> = {
    gemini_api_key: overrideStates?.gemini_api_key ?? rowState(settings, "gemini_api_key", cloud),
    groq_api_key: overrideStates?.groq_api_key ?? rowState(settings, "groq_api_key", cloud),
  };

  container.innerHTML = `
    <div class="setting-hint" style="margin-bottom: 12px;">
      Keys are stored locally in your JustSay settings file. They are never sent back in API responses.
    </div>
    ${ROWS.map((spec) => renderRowMarkup(spec, states[spec.field])).join("")}
  `;

  for (const spec of ROWS) {
    wireKey(container, spec, states[spec.field], settings, cloud, states);
  }

  for (const spec of ROWS) {
    if (states[spec.field] === "editing") {
      const input = container.querySelector<HTMLInputElement>(
        `#${prefix(spec.field)}-key-input`,
      );
      if (input) {
        requestAnimationFrame(() => input.focus());
      }
      break;
    }
  }

  return () => {};
}

function wireKey(
  container: HTMLElement,
  spec: KeyRowSpec,
  state: KeyRowState,
  settings: UserSettings,
  cloud: CloudKeyStatus | null,
  states: Record<KeyField, KeyRowState>,
): void {
  const p = prefix(spec.field);
  const input = container.querySelector<HTMLInputElement>(`#${p}-key-input`)!;
  const status = container.querySelector<HTMLElement>(`#${p}-status`)!;

  if (state === "stored" || state === "env") {
    const replaceBtn = container.querySelector<HTMLButtonElement>(`#${p}-replace`)!;
    replaceBtn.addEventListener("click", () => {
      renderKeys(container, settings, cloud, {
        ...states,
        [spec.field]: "editing",
      });
    });
    return;
  }

  if (state === "editing") {
    const cancelBtn = container.querySelector<HTMLButtonElement>(`#${p}-cancel`);
    if (cancelBtn) {
      cancelBtn.addEventListener("click", () => {
        renderKeys(container, settings, cloud, {
          ...states,
          [spec.field]: rowState(settings, spec.field, cloud),
        });
      });
    }
  }

  const saveBtn = container.querySelector<HTMLButtonElement>(`#${p}-save`)!;

  input.addEventListener("input", () => {
    saveBtn.disabled = input.value.trim() === "";
  });

  saveBtn.addEventListener("click", async () => {
    const value = input.value.trim();
    if (!value) return;

    saveBtn.disabled = true;
    input.disabled = true;
    saveBtn.textContent = "Saving…";
    status.textContent = "";
    const cancelBtnLock = container.querySelector<HTMLButtonElement>(`#${p}-cancel`);
    if (cancelBtnLock) cancelBtnLock.disabled = true;

    try {
      const { settings: fresh } = await saveSettings({
        [spec.field]: value,
      } as Partial<UserSettings>);
      renderKeys(container, fresh, getCloudKeyStatus());
    } catch (err) {
      status.textContent = `Error: ${(err as Error).message}`;
      saveBtn.disabled = false;
      input.disabled = false;
      saveBtn.textContent = "Save";
      if (cancelBtnLock) cancelBtnLock.disabled = false;
    }
  });

  input.dispatchEvent(new Event("input"));
}
