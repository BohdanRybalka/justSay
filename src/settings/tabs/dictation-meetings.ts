/**
 * MEETINGS: the Record meetings switch. The first time it is turned on, the
 * meeting disclosure (ADR 040 obligation 3) opens under the row and the switch
 * stays off until "I understand". The hint names where a meeting becomes text,
 * following the Cloud / Local choice.
 */
import { meetingsTurnedOn, type UserSettings } from "../../api";
import { saveSettings, type TabLifecycle } from "../settings";
import { emitSettingsChanged } from "./dictation";
import { notifyError } from "../../notify";
import { renderToggle } from "../../ui/controls";
import { icon } from "../../ui/icons";

type Mode = UserSettings["stt_mode"];

const WHERE_IT_BECOMES_TEXT: Readonly<Record<Mode, string>> = {
  cloud: "in the cloud",
  local: "on this computer",
};

export function meetingsHint(mode: Mode): string {
  return `Right-click the widget and pick Record a meeting · turned into text ${WHERE_IT_BECOMES_TEXT[mode]}`;
}

export interface MeetingsGroup extends TabLifecycle {
  showMode: (mode: Mode) => void;
}

/** Adds the MEETINGS group to the end of `container`, the switch showing
 *  whether the backend will start a meeting recording. */
export function renderDictationMeetings(container: HTMLElement, settings: UserSettings): MeetingsGroup {
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
          In Cloud mode the other participants' audio is sent to the transcription provider
          you configured, along with your own. Switch to Local mode if none of it may leave
          this machine.
        </p>
        <button type="button" class="btn btn-primary btn-small" id="btn-meeting-consent">I understand</button>
      </div>
    </div>
  `,
  );

  const hint = container.querySelector<HTMLElement>("#meetings-hint")!;
  const toggle = container.querySelector<HTMLButtonElement>("#meetings-toggle")!;
  const disclosure = container.querySelector<HTMLElement>("#meeting-disclosure")!;
  const consentButton = container.querySelector<HTMLButtonElement>("#btn-meeting-consent")!;

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

  function showMode(mode: Mode) {
    hint.textContent = meetingsHint(mode);
  }

  showMode(settings.stt_mode);

  return {
    destroy: () => {
      destroyed = true;
    },
    showMode,
  };
}
