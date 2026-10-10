/**
 * The Meetings page: the card that starts and stops a meeting, then LANGUAGE
 * (how a meeting is heard: detected for each part, or one picked from the
 * list) and MODEL (where a stopped meeting becomes text, apart from dictation;
 * the card's hint names it).
 */
import type { UserSettings } from "../../api";
import { getCloudKeyStatus, saveSettings, type TabLifecycle } from "../settings";
import { emitSettingsChanged } from "./dictation";
import { CLOUD_KEY_MISSING, cloudKeyMissing, drawRow, modeRowHtml } from "./mode-rows";
import { notifyError } from "../../notify";
import { mountMeetingCard } from "./meeting-card";
import { renderSelect, renderToggle } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { DICTATION_LANGUAGES } from "../../languages";

type Engine = UserSettings["meetings_engine"];

const DETECT = "auto";

const CLOUD_HINT = "Everyone's voices go to your cloud provider";
const LOCAL_HINT = "Stays on this computer";

/** Adds the Meetings page to the end of `container`; `windowHidden` keeps the
 *  card from polling until the window is shown. */
export function renderMeetings(
  container: HTMLElement,
  settings: UserSettings,
  windowHidden: boolean,
): TabLifecycle {
  let language = settings.meetings_language;
  let picked = language === DETECT ? settings.language : language;

  container.insertAdjacentHTML(
    "beforeend",
    `
    <h2 class="panel-title">Meetings</h2>
    <p class="panel-subtitle">Record a call in any app and get its text.</p>
    <div id="meeting-card"></div>
    <div class="group-label">${icon("globe")}LANGUAGE</div>
    <div class="card">
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Detect the language</div>
          <div class="setting-row-hint">For calls in several languages</div>
        </div>
        <div class="setting-row-controls">
          <button id="meetings-detect" aria-label="Detect the language"></button>
        </div>
      </div>
      <div class="setting-row" id="meetings-language-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Meeting language</div>
        </div>
        <div class="setting-row-controls">
          <select id="meetings-language" aria-label="Meeting language">
            ${DICTATION_LANGUAGES.map(
              (l) => `<option value="${l.code}" ${l.code === picked ? "selected" : ""}>${l.label}</option>`,
            ).join("")}
          </select>
        </div>
      </div>
    </div>
    <div class="group-label">${icon("chip")}MODEL</div>
    <div class="card" role="radiogroup" aria-label="Meeting model">
      ${modeRowHtml("meetings-cloud", "cloud", "Cloud")}
      ${modeRowHtml("meetings-local", "chip", "Local model")}
    </div>
  `,
  );

  const card = mountMeetingCard(
    container.querySelector<HTMLElement>("#meeting-card")!,
    settings,
    windowHidden,
  );
  const detectToggle = container.querySelector<HTMLButtonElement>("#meetings-detect")!;
  const languageRow = container.querySelector<HTMLElement>("#meetings-language-row")!;
  const languageSelect = container.querySelector<HTMLSelectElement>("#meetings-language")!;
  const cloudRow = container.querySelector<HTMLButtonElement>("#meetings-cloud")!;
  const localRow = container.querySelector<HTMLButtonElement>("#meetings-local")!;
  const keyMissing = cloudKeyMissing(getCloudKeyStatus());

  let destroyed = false;

  async function save(updates: Partial<UserSettings>): Promise<boolean> {
    try {
      await saveSettings(updates);
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
      return false;
    }
    void emitSettingsChanged();
    return true;
  }

  function showLanguage() {
    detectToggle.setAttribute("aria-checked", String(language === DETECT));
    languageRow.hidden = language === DETECT;
    languageSelect.value = picked;
  }

  async function chooseLanguage(next: string) {
    languageRow.hidden = next === DETECT;
    const saved = await save({ meetings_language: next });
    if (destroyed) return;
    if (saved) {
      language = next;
      if (next !== DETECT) picked = next;
    }
    showLanguage();
  }

  renderToggle(detectToggle, language === DETECT, (detect) => void chooseLanguage(detect ? DETECT : picked));
  renderSelect(languageSelect);
  languageSelect.addEventListener("change", () => void chooseLanguage(languageSelect.value));
  showLanguage();

  let engine = settings.meetings_engine;

  let savingEngine = false;

  function showEngine(shown: Engine) {
    card.setEngine(shown);
    drawRow(cloudRow, shown === "cloud", {
      hint: keyMissing ? CLOUD_KEY_MISSING : CLOUD_HINT,
      alert: keyMissing,
      locked: false,
      disabled: savingEngine,
      action: "",
    });
    drawRow(localRow, shown === "local", {
      hint: LOCAL_HINT,
      alert: false,
      locked: false,
      disabled: savingEngine,
      action: "",
    });
  }

  async function chooseEngine(next: Engine) {
    if (savingEngine || next === engine) return;
    savingEngine = true;
    showEngine(next);
    const saved = await save({ meetings_engine: next });
    if (destroyed) return;
    savingEngine = false;
    if (saved) engine = next;
    showEngine(engine);
  }

  cloudRow.addEventListener("click", () => void chooseEngine("cloud"));
  localRow.addEventListener("click", () => void chooseEngine("local"));
  showEngine(engine);

  return {
    destroy: () => {
      destroyed = true;
      card.destroy();
    },
    releaseResources: card.releaseResources,
    resumeResources: card.resumeResources,
  };
}
