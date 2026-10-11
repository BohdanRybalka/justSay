/**
 * LANGUAGE on a page that hears its own recordings (Meetings, Files): Detect the language, and
 * with it off the one language to hear them in. Detection is stored as "auto"; turning it off
 * comes back to the language picked last, which starts as the dictation language.
 */
import type { UserSettings } from "../../api";
import { DICTATION_LANGUAGES } from "../../languages";
import { renderSelect, renderToggle } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { saveChoice } from "./save-choice";

const DETECT = "auto";

export interface LanguageCardSpec {
  /** Prefix of the element ids: `<prefix>-detect`, `<prefix>-language-row`, `<prefix>-language`. */
  idPrefix: string;
  field: "meetings_language" | "files_language";
  detectHint: string;
  listTitle: string;
}

/** Adds the LANGUAGE group to the end of `container`. `destroy` keeps a save still in flight
 *  from drawing on a page that is gone. */
export function mountLanguageCard(
  container: HTMLElement,
  settings: UserSettings,
  spec: LanguageCardSpec,
): { destroy: () => void } {
  const { idPrefix, field } = spec;
  let language = settings[field];
  let picked = language === DETECT ? settings.language : language;

  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("globe")}LANGUAGE</div>
    <div class="card">
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Detect the language</div>
          <div class="setting-row-hint">${spec.detectHint}</div>
        </div>
        <div class="setting-row-controls">
          <button id="${idPrefix}-detect" aria-label="Detect the language"></button>
        </div>
      </div>
      <div class="setting-row" id="${idPrefix}-language-row">
        <div class="setting-row-text">
          <div class="setting-row-title">${spec.listTitle}</div>
        </div>
        <div class="setting-row-controls">
          <select id="${idPrefix}-language" aria-label="${spec.listTitle}">
            ${DICTATION_LANGUAGES.map(
              (l) => `<option value="${l.code}" ${l.code === picked ? "selected" : ""}>${l.label}</option>`,
            ).join("")}
          </select>
        </div>
      </div>
    </div>
  `,
  );

  const detectToggle = container.querySelector<HTMLButtonElement>(`#${idPrefix}-detect`)!;
  const languageRow = container.querySelector<HTMLElement>(`#${idPrefix}-language-row`)!;
  const languageSelect = container.querySelector<HTMLSelectElement>(`#${idPrefix}-language`)!;
  let destroyed = false;

  function show() {
    detectToggle.setAttribute("aria-checked", String(language === DETECT));
    languageRow.hidden = language === DETECT;
    languageSelect.value = picked;
  }

  async function choose(next: string) {
    languageRow.hidden = next === DETECT;
    const saved = await saveChoice({ [field]: next });
    if (destroyed) return;
    if (saved) {
      language = next;
      if (next !== DETECT) picked = next;
    }
    show();
  }

  renderToggle(detectToggle, language === DETECT, (detect) => void choose(detect ? DETECT : picked));
  renderSelect(languageSelect);
  languageSelect.addEventListener("change", () => void choose(languageSelect.value));
  show();

  return {
    destroy: () => {
      destroyed = true;
    },
  };
}
