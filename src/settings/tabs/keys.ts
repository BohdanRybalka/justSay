/**
 * The API keys fold: one row per cloud key and the row that picks where
 * recordings go. A key row is stored, env, unset, unknown or editing, and is
 * redrawn alone; each key's subtitle follows the routing choice, and in Local
 * mode says the key is not used.
 */
import { type CloudKeyStatus, type UserSettings } from "../../api";
import { MASKED_API_KEY } from "../../contracts";
import { notifyError } from "../../notify";
import { renderSegmented, type SegmentedOption } from "../../ui/controls";
import { saveSettings, getCloudKeyStatus } from "../settings";

type KeyField = "gemini_api_key" | "groq_api_key";
type KeyRowState = "stored" | "env" | "unset" | "unknown" | "editing";
type Engine = UserSettings["stt_engine"];
type Provider = Exclude<Engine, "auto">;

interface KeyRowSpec {
  field: KeyField;
  provider: Provider;
  label: string;
}

const ROWS: readonly KeyRowSpec[] = [
  { field: "groq_api_key", provider: "groq", label: "Groq" },
  { field: "gemini_api_key", provider: "gemini", label: "Google" },
];

const ENGINES: readonly SegmentedOption<Engine>[] = [
  { value: "auto", label: "Automatic" },
  { value: "groq", label: "Groq" },
  { value: "gemini", label: "Google" },
];

const ROUTE_HINTS: Readonly<Record<Engine, Readonly<Record<Provider, string>>>> = {
  auto: { groq: "Used for short recordings", gemini: "Used for long recordings" },
  groq: { groq: "Used for all recordings", gemini: "Used for files Groq can't read" },
  gemini: { groq: "Not used for recordings", gemini: "Used for all recordings" },
};

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

function routeHint(settings: UserSettings, engine: Engine, provider: Provider): string {
  return settings.stt_mode === "local" ? LOCAL_HINT : ROUTE_HINTS[engine][provider];
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

/**
 * Fills `container` with the Groq, Google and "Recordings go to" rows. Save
 * and the routing choice go through `saveSettings`; a routing save that fails
 * puts the stored choice back and says so, unless a newer choice was made since.
 */
export function renderKeys(container: HTMLElement, settings: UserSettings, cloud: CloudKeyStatus | null): void {
  let current = settings;
  let knownCloud = cloud;
  container.innerHTML = `
    ${ROWS.map((spec) => `<div class="setting-row key-row" data-provider="${spec.provider}"></div>`).join("")}
    <div class="setting-row">
      <div class="setting-row-text"><div class="setting-row-title">Recordings go to</div></div>
      <div class="setting-row-controls"><div class="route-choice" aria-label="Recordings go to"></div></div>
    </div>
  `;
  const rowOf = (spec: KeyRowSpec): HTMLElement =>
    container.querySelector<HTMLElement>(`.key-row[data-provider="${spec.provider}"]`)!;

  const showRoutes = (engine: Engine): void => {
    for (const spec of ROWS) {
      rowOf(spec).querySelector<HTMLElement>(".route-hint")!.textContent = routeHint(current, engine, spec.provider);
    }
  };

  const draw = (spec: KeyRowSpec, state: KeyRowState, refocus?: boolean): void => {
    const row = rowOf(spec);
    const hadFocus = refocus ?? row.contains(document.activeElement);
    row.innerHTML = `
      <div class="setting-row-text">
        <div class="setting-row-title">${spec.label}</div>
        <div class="setting-row-hint route-hint">${routeHint(current, current.stt_engine, spec.provider)}</div>
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

  const choice = container.querySelector<HTMLElement>(".route-choice")!;
  const showChoice = (engine: Engine): void =>
    renderSegmented(choice, ENGINES, engine, (next) => void chooseEngine(next));
  let latestChoice = 0;
  const chooseEngine = async (engine: Engine): Promise<void> => {
    const token = ++latestChoice;
    showRoutes(engine);
    try {
      const { settings: fresh } = await saveSettings({ stt_engine: engine });
      if (token === latestChoice) current = fresh;
    } catch (e) {
      if (token !== latestChoice) return;
      showChoice(current.stt_engine);
      showRoutes(current.stt_engine);
      void notifyError(`Could not save where recordings go: ${describeFailure(e)}`);
    }
  };

  for (const spec of ROWS) draw(spec, rowState(current, spec.field, knownCloud));
  showChoice(current.stt_engine);
}
