import { api } from "../api";
import {
  EVENT_MEETING_TOGGLE,
  EVENT_NAVIGATE_PANEL,
  EVENT_RING_CLOSED,
  EVENT_RING_OPENED,
  EVENT_RING_POINTER,
  EVENT_SETTINGS_CHANGED,
  type NavigatePanel,
  type RingPointer,
} from "../contracts";
import { mountIconSprite } from "../ui/icons";
import { applyThemePreference } from "../ui/theme";
import { petalLabel, stepFor, UNREAD_RING_STATE, type RingAction, type RingState } from "./ring-actions";
import { closeRing, followPointer, mountRing, openRing, renderPetalLabels } from "./ring-view";

const ring = document.getElementById("ring")!;
let state: RingState = UNREAD_RING_STATE;

mountIconSprite(document);
applyThemePreference("system");
mountRing(ring, (action) => void askShellToClose().then(() => carryOut(action)));
renderPetalLabels(ring, (action) => petalLabel(action, state));
document.addEventListener("contextmenu", (event) => event.preventDefault());

async function askShellToClose() {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("close_ring");
  } catch (e) {
    console.warn("Closing the ring failed:", e);
  }
}

async function openPanel(target: NavigatePanel) {
  const { emit } = await import("@tauri-apps/api/event");
  await emit(EVENT_NAVIGATE_PANEL, target);
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("show_settings_window");
}

async function switchLanguage(language: string) {
  await api.updateSettings({ language });
  const { emit } = await import("@tauri-apps/api/event");
  await emit(EVENT_SETTINGS_CHANGED);
}

async function carryOut(action: RingAction) {
  const step = stepFor(action, state);
  try {
    if (step.kind === "toggle-meeting") {
      const { emit } = await import("@tauri-apps/api/event");
      await emit(EVENT_MEETING_TOGGLE);
    } else if (step.kind === "pick-file") {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("pick_audio_file");
    } else if (step.kind === "switch-language") {
      await switchLanguage(step.language);
    } else {
      await openPanel(step.target);
    }
  } catch (e) {
    console.warn(`The ring's ${action} petal failed:`, e);
  }
}

/** Settings and the meeting state are read apart, so a failed meeting read
 *  still leaves the theme and the language label current. */
async function readState() {
  const [settings, meeting] = await Promise.allSettled([api.getSettings(), api.getMeetingStatus()]);
  if (settings.status === "fulfilled") applyThemePreference(settings.value.theme);
  else console.warn("Could not read the settings for the ring:", settings.reason);
  state = {
    settings: settings.status === "fulfilled" ? settings.value : state.settings,
    meetingRecording: meeting.status === "fulfilled" ? meeting.value.is_recording : state.meetingRecording,
  };
  renderPetalLabels(ring, (action) => petalLabel(action, state));
}

async function listenToShell() {
  try {
    const { listen } = await import("@tauri-apps/api/event");
    await listen(EVENT_RING_OPENED, () => {
      openRing(ring);
      void readState();
    });
    await listen(EVENT_RING_CLOSED, () => closeRing(ring));
    await listen<RingPointer>(EVENT_RING_POINTER, ({ payload }) => followPointer(ring, payload));
    await listen(EVENT_SETTINGS_CHANGED, () => void readState());
  } catch (e) {
    console.warn("The ring cannot hear the shell:", e);
  }
}

void readState();
void listenToShell();
