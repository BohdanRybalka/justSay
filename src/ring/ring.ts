import { api } from "../api";
import {
  EVENT_RING_CLOSED,
  EVENT_RING_OPENED,
  EVENT_RING_POINTER,
  EVENT_SETTINGS_CHANGED,
  type RingPointer,
} from "../contracts";
import { mountIconSprite } from "../ui/icons";
import { applyThemePreference } from "../ui/theme";
import { closeRing, followPointer, mountRing, openRing } from "./ring-view";

const ring = document.getElementById("ring")!;

mountIconSprite(document);
applyThemePreference("system");
mountRing(ring, () => void askShellToClose());
document.addEventListener("contextmenu", (event) => event.preventDefault());

async function askShellToClose() {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("close_ring");
  } catch (e) {
    console.warn("Closing the ring failed:", e);
  }
}

async function applyThemeSetting() {
  try {
    applyThemePreference((await api.getSettings()).theme);
  } catch (e) {
    console.warn("Could not read the theme for the ring:", e);
  }
}

async function listenToShell() {
  try {
    const { listen } = await import("@tauri-apps/api/event");
    await listen(EVENT_RING_OPENED, () => {
      openRing(ring);
      void applyThemeSetting();
    });
    await listen(EVENT_RING_CLOSED, () => closeRing(ring));
    await listen<RingPointer>(EVENT_RING_POINTER, ({ payload }) => followPointer(ring, payload));
    await listen(EVENT_SETTINGS_CHANGED, () => void applyThemeSetting());
  } catch (e) {
    console.warn("The ring cannot hear the shell:", e);
  }
}

void applyThemeSetting();
void listenToShell();
