/**
 * MEETINGS: the Record meetings switch. The first time it is turned on, the
 * meeting disclosure (ADR 040 obligation 3) opens under the row and the switch
 * stays off until "I understand". Below it, the language a meeting is heard in:
 * detected for each part, or one picked from the list. MEETING MODEL then picks
 * where a stopped meeting becomes text, apart from dictation (the hint names it).
 */
import { meetingsTurnedOn, type UserSettings } from "../../api";
import { getCloudKeyStatus, saveSettings, type TabLifecycle } from "../settings";
import { emitSettingsChanged } from "./dictation";
import { CLOUD_KEY_MISSING, cloudKeyMissing, drawRow, modeRowHtml } from "./mode-rows";
import { notifyError } from "../../notify";
import { renderSelect, renderToggle } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { DICTATION_LANGUAGES } from "../../languages";

type Engine = UserSettings["meetings_engine"];

const DETECT = "auto";

const WHERE_IT_BECOMES_TEXT: Readonly<Record<Engine, string>> = {
  cloud: "in the cloud",
  local: "on this computer",
};

const CLOUD_HINT = "Everyone's voices go to your cloud provider";
const LOCAL_HINT = "Stays on this computer";

export function meetingsHint(engine: Engine): string {
  return `Right-click the widget and pick Record a meeting · turned into text ${WHERE_IT_BECOMES_TEXT[engine]}`;
}

/** Adds the MEETINGS and MEETING MODEL groups to the end of `container`, the
 *  switch showing whether the backend will start a meeting recording. */
export function renderDictationMeetings(container: HTMLElement, settings: UserSettings): TabLifecycle {
  let language = settings.meetings_language;
  let picked = language === DETECT ? settings.language : language;

  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("users")}MEETINGS</div>
    <div class="card">
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Record meetings<span class="chip">new</span></div>
          <div class="setting-row-hint" id="meetings-hint"></div>
        </div>
        <div class="setting-row-controls">
          <button id="meetings-toggle" aria-label="Record meetings"></button>
        </div>
      </div>
      <div class="meeting-disclosure" id="meeting-disclosure" hidden>
        <p id="meeting-consent-responsibility">
          A meeting recording captures everyone on the call, including people who never
          installed JustSay. You are responsible for obtaining whatever consent your
          jurisdiction and your employer require before you start one.
        </p>
        <p id="meeting-consent-cloud">
          When meetings are turned into text in the cloud, the other participants' audio is
          sent to the transcription provider you configured, along with your own. Keep them on
          this computer if none of it may leave this machine.
        </p>
        <button type="button" class="btn btn-primary btn-small" id="btn-meeting-consent">I understand</button>
      </div>
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
    <div class="group-label">${icon("chip")}MEETING MODEL</div>
    <div class="card" role="radiogroup" aria-label="Meeting model">
      ${modeRowHtml("meetings-cloud", "cloud", "Cloud")}
      ${modeRowHtml("meetings-local", "chip", "Local model")}
    </div>
  `,
  );

  const hint = container.querySelector<HTMLElement>("#meetings-hint")!;
  const toggle = container.querySelector<HTMLButtonElement>("#meetings-toggle")!;
  const disclosure = container.querySelector<HTMLElement>("#meeting-disclosure")!;
  const consentButton = container.querySelector<HTMLButtonElement>("#btn-meeting-consent")!;
  const detectToggle = container.querySelector<HTMLButtonElement>("#meetings-detect")!;
  const languageRow = container.querySelector<HTMLElement>("#meetings-language-row")!;
  const languageSelect = container.querySelector<HTMLSelectElement>("#meetings-language")!;
  const cloudRow = container.querySelector<HTMLButtonElement>("#meetings-cloud")!;
  const localRow = container.querySelector<HTMLButtonElement>("#meetings-local")!;
  const keyMissing = cloudKeyMissing(getCloudKeyStatus());

  let acknowledged = settings.meeting_consent_acknowledged;
  let on = meetingsTurnedOn(settings);
  let destroyed = false;

  function drawSwitch() {
    toggle.setAttribute("aria-checked", String(on));
  }

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

  async function turn(next: boolean) {
    toggle.disabled = true;
    const saved = await save({ meetings_enabled: next });
    if (destroyed) return;
    toggle.disabled = false;
    if (saved) on = next;
    drawSwitch();
  }

  renderToggle(toggle, on, (next) => {
    if (next && !acknowledged) {
      drawSwitch();
      disclosure.hidden = false;
      return;
    }
    void turn(next);
  });

  consentButton.addEventListener("click", async () => {
    consentButton.disabled = true;
    const saved = await save({ meeting_consent_acknowledged: true, meetings_enabled: true });
    if (destroyed) return;
    consentButton.disabled = false;
    if (!saved) return;
    acknowledged = true;
    on = true;
    disclosure.hidden = true;
    drawSwitch();
  });

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

  function showEngine(shown: Engine) {
    hint.textContent = meetingsHint(shown);
    drawRow(cloudRow, shown === "cloud", {
      hint: keyMissing ? CLOUD_KEY_MISSING : CLOUD_HINT,
      alert: keyMissing,
      locked: false,
      disabled: false,
      action: "",
    });
    drawRow(localRow, shown === "local", { hint: LOCAL_HINT, alert: false, locked: false, disabled: false, action: "" });
  }

  async function chooseEngine(next: Engine) {
    if (next === engine) return;
    showEngine(next);
    const saved = await save({ meetings_engine: next });
    if (destroyed) return;
    if (saved) engine = next;
    showEngine(engine);
  }

  cloudRow.addEventListener("click", () => void chooseEngine("cloud"));
  localRow.addEventListener("click", () => void chooseEngine("local"));
  showEngine(engine);

  return {
    destroy: () => {
      destroyed = true;
    },
  };
}
