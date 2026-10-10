/**
 * The card at the top of Meetings. It captures nothing itself: the widget owns
 * the capture, so a press goes to it as `EVENT_MEETING_TOGGLE`, and the card
 * shows what the backend reports once a second while the window is on screen.
 * A first start turns the card into the meeting disclosure (ADR 040 obligation
 * 3); the toggle is sent only after "I understand" is saved.
 */
import { api, meetingLevelStream, type MeetingStatus, type UserSettings } from "../../api";
import { EVENT_MEETING_TOGGLE } from "../../contracts";
import { loadEventApi } from "../../event-api";
import { formatElapsedClock } from "../../format";
import { levelFromDb } from "../../level";
import { notifyError } from "../../notify";
import { icon } from "../../ui/icons";
import { saveSettings, type TabLifecycle } from "../settings";
import { emitSettingsChanged } from "./dictation";

type Engine = UserSettings["meetings_engine"];
type View = "start" | "disclosure" | "recording";

const POLL_MS = 1000;
const PRESS_SETTLE_MS = 5000;
const CLOCK_DRIFT_MS = 2000;

const WHERE_IT_BECOMES_TEXT: Readonly<Record<Engine, string>> = {
  cloud: "in the cloud",
  local: "on this computer",
};

export function meetingCardHint(engine: Engine): string {
  return `Your microphone and what this computer plays · turned into text ${WHERE_IT_BECOMES_TEXT[engine]}`;
}

export interface MeetingCard extends TabLifecycle {
  setEngine: (engine: Engine) => void;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Fills `card` and keeps it current; `windowHidden` says whether polling may start now. */
export function mountMeetingCard(
  card: HTMLElement,
  settings: UserSettings,
  windowHidden: boolean,
): MeetingCard {
  let engine = settings.meetings_engine;
  let acknowledged = settings.meeting_consent_acknowledged;
  let disclosureOpen = false;
  let recording = false;
  let statusKnown = false;
  let startedAt = 0;
  let pressedWhileRecording: boolean | null = null;
  let pressTimer: ReturnType<typeof setTimeout> | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let levels: AbortController | null = null;
  let latestRead = 0;
  let reading = false;
  let destroyed = false;

  function currentView(): View {
    if (recording) return "recording";
    return disclosureOpen ? "disclosure" : "start";
  }

  function startHtml(): string {
    return `
      <span class="brand-mark">${icon("users")}</span>
      <span class="meeting-card-text">
        <b>Record a meeting</b>
        <small id="meeting-hint">${meetingCardHint(engine)}</small>
      </span>
      <button type="button" class="btn btn-primary" id="meeting-start">${icon("mic", "small")}Start recording</button>
    `;
  }

  function disclosureHtml(): string {
    return `
      <div class="meeting-disclosure">
        <b>Before your first meeting</b>
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
        <button type="button" class="btn btn-primary btn-small" id="meeting-accept">I understand, start recording</button>
        <button type="button" class="btn btn-small" id="meeting-cancel">Cancel</button>
      </div>
    `;
  }

  function recordingHtml(): string {
    return `
      <span class="brand-mark">${icon("users")}</span>
      <span class="meeting-card-text">
        <b>Recording · <span class="num" id="meeting-timer"></span></b>
        <span class="voices">
          <span class="voice">${icon("mic", "small")}You<span class="level-meter"><i id="meeting-level-mic"></i></span></span>
          <span class="voice">${icon("speaker", "small")}Others<span class="level-meter"><i id="meeting-level-system"></i></span></span>
        </span>
      </span>
      <button type="button" class="btn" id="meeting-stop">${icon("stop", "small")}Stop</button>
    `;
  }

  function draw() {
    const view = currentView();
    card.className = `card meeting-card meeting-card--${view}`;
    card.innerHTML =
      view === "recording" ? recordingHtml() : view === "disclosure" ? disclosureHtml() : startHtml();
    if (pressedWhileRecording !== null || !statusKnown) disableButtons();
    card.querySelector("#meeting-start")?.addEventListener("click", begin);
    card.querySelector("#meeting-accept")?.addEventListener("click", () => void accept());
    card.querySelector("#meeting-cancel")?.addEventListener("click", cancel);
    card.querySelector("#meeting-stop")?.addEventListener("click", () => void requestToggle());
    drawTimer();
  }

  function disableButtons() {
    card.querySelectorAll<HTMLButtonElement>("button").forEach((button) => (button.disabled = true));
  }

  function drawTimer() {
    const timer = card.querySelector("#meeting-timer");
    if (timer) timer.textContent = formatElapsedClock((Date.now() - startedAt) / 1000);
  }

  function releasePress() {
    if (pressTimer !== null) clearTimeout(pressTimer);
    pressTimer = null;
    pressedWhileRecording = null;
    if (!destroyed) draw();
  }

  async function requestToggle() {
    const shownAtPress = recording;
    pressedWhileRecording = shownAtPress;
    if (pressTimer !== null) clearTimeout(pressTimer);
    pressTimer = setTimeout(releasePress, PRESS_SETTLE_MS);
    draw();
    const current = await api.getMeetingStatus().catch(() => null);
    if (current !== null && !destroyed) applyStatus(current);
    if (recording !== shownAtPress) {
      releasePress();
      return;
    }
    try {
      const { emit } = await loadEventApi();
      await emit(EVENT_MEETING_TOGGLE);
    } catch (e) {
      notifyError(errorText(e));
      releasePress();
    }
  }

  function begin() {
    if (acknowledged) {
      void requestToggle();
      return;
    }
    disclosureOpen = true;
    draw();
  }

  function cancel() {
    disclosureOpen = false;
    draw();
  }

  async function accept() {
    disableButtons();
    try {
      await saveSettings({ meeting_consent_acknowledged: true });
    } catch (e) {
      notifyError(errorText(e));
      if (!destroyed) draw();
      return;
    }
    acknowledged = true;
    disclosureOpen = false;
    void emitSettingsChanged();
    await requestToggle();
  }

  function showLevels(stream: AbortController, mic: number, system: number) {
    if (levels !== stream) return;
    const micFill = card.querySelector<HTMLElement>("#meeting-level-mic");
    const systemFill = card.querySelector<HTMLElement>("#meeting-level-system");
    if (micFill) micFill.style.width = `${mic * 100}%`;
    if (systemFill) systemFill.style.width = `${system * 100}%`;
  }

  function openLevels() {
    if (levels) return;
    const stream: AbortController = meetingLevelStream(
      (data) => showLevels(stream, levelFromDb(data.mic_db), levelFromDb(data.system_db)),
      () => dropLevels(stream),
      (error) => {
        console.warn("The meeting level stream stopped:", error);
        dropLevels(stream);
      },
    );
    levels = stream;
  }

  function dropLevels(stream: AbortController) {
    showLevels(stream, 0, 0);
    if (levels === stream) levels = null;
  }

  function closeLevels() {
    levels?.abort();
    levels = null;
  }

  function applyStatus(status: MeetingStatus) {
    const wasRecording = recording;
    const wasKnown = statusKnown;
    statusKnown = true;
    recording = status.is_recording;
    const elapsedMs = status.duration_seconds * 1000;
    if (recording && (!wasRecording || Math.abs(Date.now() - startedAt - elapsedMs) > CLOCK_DRIFT_MS)) {
      startedAt = Date.now() - elapsedMs;
    }
    if (pressedWhileRecording !== null && recording !== pressedWhileRecording) {
      pressedWhileRecording = null;
      if (pressTimer !== null) clearTimeout(pressTimer);
      pressTimer = null;
    }
    if (recording !== wasRecording || !wasKnown) {
      disclosureOpen = false;
      draw();
    }
    if (recording) openLevels();
    else closeLevels();
  }

  async function read() {
    if (reading) return;
    reading = true;
    const token = latestRead;
    const status = await api.getMeetingStatus().catch(() => null);
    reading = false;
    if (status === null || destroyed || token !== latestRead) return;
    applyStatus(status);
  }

  function resumeResources() {
    if (pollTimer === null) {
      pollTimer = setInterval(() => {
        drawTimer();
        void read();
      }, POLL_MS);
    }
    void read();
  }

  function releaseResources() {
    if (pollTimer !== null) clearInterval(pollTimer);
    pollTimer = null;
    latestRead += 1;
    closeLevels();
  }

  draw();
  if (!windowHidden) resumeResources();

  return {
    setEngine: (next) => {
      engine = next;
      const hint = card.querySelector("#meeting-hint");
      if (hint) hint.textContent = meetingCardHint(engine);
    },
    releaseResources,
    resumeResources,
    destroy: () => {
      destroyed = true;
      if (pressTimer !== null) clearTimeout(pressTimer);
      releaseResources();
    },
  };
}
