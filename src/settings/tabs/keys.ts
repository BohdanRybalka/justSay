import { api, type UserSettings } from "../../api";

type KeyField = "gemini_api_key" | "groq_api_key";
type KeyRowState = "stored" | "unset" | "editing";

interface KeyRowSpec {
  field: KeyField;
  label: string;
  unsetHint: string;
  storedHint: string;
  placeholder: string;
}

const ROWS: KeyRowSpec[] = [
  {
    field: "gemini_api_key",
    label: "Gemini (STT — long audio &amp; structured)",
    unsetHint: "No key set — cloud STT will fail.",
    storedHint: "Key stored.",
    placeholder: "Paste your Gemini API key",
  },
  {
    field: "groq_api_key",
    label: "Groq (STT — short audio &amp; LLM)",
    unsetHint: "No key set — cloud STT and LLM will fail.",
    storedHint: "Key stored.",
    placeholder: "Paste your Groq API key",
  },
];

const MASKED_DISPLAY = "••••••••";

function prefix(field: KeyField): string {
  return field === "gemini_api_key" ? "gemini" : "groq";
}

function rowState(settings: UserSettings, field: KeyField): KeyRowState {
  return settings[field] === "***" ? "stored" : "unset";
}

function renderRowMarkup(spec: KeyRowSpec, state: KeyRowState): string {
  const p = prefix(spec.field);
  let inner: string;
  if (state === "stored") {
    inner = `
      <input type="text" id="${p}-key-input" disabled
        value="${MASKED_DISPLAY}"
        aria-label="Stored API key (masked)" />
      <div class="key-actions">
        <button class="btn btn-secondary btn-sm" id="${p}-replace">Replace</button>
      </div>
    `;
  } else if (state === "editing") {
    // Editing entered via Replace — show Cancel to abort the change.
    inner = `
      <input type="password" id="${p}-key-input" autocomplete="off" spellcheck="false"
        placeholder="${spec.placeholder}" />
      <div class="key-actions">
        <button class="btn btn-secondary btn-sm" id="${p}-cancel">Cancel</button>
        <button class="btn btn-primary btn-sm" id="${p}-save" disabled>Save</button>
      </div>
    `;
  } else {
    // unset — first-time entry; no prior state to cancel back to.
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
  overrideStates?: Partial<Record<KeyField, KeyRowState>>,
): () => void {
  const states: Record<KeyField, KeyRowState> = {
    gemini_api_key: overrideStates?.gemini_api_key ?? rowState(settings, "gemini_api_key"),
    groq_api_key: overrideStates?.groq_api_key ?? rowState(settings, "groq_api_key"),
  };

  container.innerHTML = `
    <div class="setting-hint" style="margin-bottom: 12px;">
      Keys are stored in <code>~/.justsay/settings.json</code>. They are never sent back in API responses.
    </div>
    ${ROWS.map((spec) => renderRowMarkup(spec, states[spec.field])).join("")}
  `;

  for (const spec of ROWS) {
    wireKey(container, spec, states[spec.field]);
  }

  // Editing-state focus: after the DOM is rebuilt, land the caret in the
  // first editing row's input so the keyboard works immediately. Wrapping
  // in rAF avoids WebView2 occasionally dropping the focus call when it
  // fires before layout has committed the rebuilt input.
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
): void {
  const p = prefix(spec.field);
  const input = container.querySelector<HTMLInputElement>(`#${p}-key-input`)!;
  const status = container.querySelector<HTMLElement>(`#${p}-status`)!;

  if (state === "stored") {
    const replaceBtn = container.querySelector<HTMLButtonElement>(`#${p}-replace`)!;
    replaceBtn.addEventListener("click", () => {
      // Single uniform update strategy: full re-render with editing state.
      renderKeys(container, snapshotSettings(container), {
        [spec.field]: "editing",
      } as Partial<Record<KeyField, KeyRowState>>);
    });
    return;
  }

  // editing has an extra Cancel button next to Save.
  if (state === "editing") {
    const cancelBtn = container.querySelector<HTMLButtonElement>(`#${p}-cancel`);
    if (cancelBtn) {
      cancelBtn.addEventListener("click", () => {
        // Abort the edit — return to stored state without touching the backend.
        renderKeys(container, snapshotSettings(container), {
          [spec.field]: "stored",
        } as Partial<Record<KeyField, KeyRowState>>);
      });
    }
  }

  // unset AND editing share the Save flow.
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
    // Lock Cancel too — without this a fast Cancel-during-Save would race
    // the in-flight `api.updateSettings` re-render and produce flicker.
    const cancelBtnLock = container.querySelector<HTMLButtonElement>(`#${p}-cancel`);
    if (cancelBtnLock) cancelBtnLock.disabled = true;

    try {
      const resp = await api.updateSettings({
        [spec.field]: value,
      } as Partial<UserSettings>);
      renderKeys(container, resp.settings);
    } catch (err) {
      status.textContent = `Error: ${(err as Error).message}`;
      saveBtn.disabled = false;
      input.disabled = false;
      saveBtn.textContent = "Save";
      if (cancelBtnLock) cancelBtnLock.disabled = false;
    }
  });

  // Activate Save on any pre-existing non-empty input
  input.dispatchEvent(new Event("input"));
}

// Snapshot the current `settings` state from the container DOM. Reading the
// masked vs editable input is enough to know whether a key is stored or
// not. Used by the Replace handler so the re-render keeps the OTHER row in
// its current state when only one row transitions to editing.
function snapshotSettings(container: HTMLElement): UserSettings {
  const isStored = (p: string): boolean => {
    const el = container.querySelector<HTMLInputElement>(`#${p}-key-input`);
    return !!el && el.disabled;
  };
  // Other UserSettings fields aren't read inside renderKeys; pass minimal
  // shape with only the two key fields populated meaningfully.
  return {
    gemini_api_key: isStored("gemini") ? "***" : "",
    groq_api_key: isStored("groq") ? "***" : "",
  } as UserSettings;
}
