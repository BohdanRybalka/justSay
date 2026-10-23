/**
 * The Meetings page: the card that starts and stops a meeting, the latest meetings, then LANGUAGE
 * (how a meeting is heard: detected for each part, or one picked from the list). Where a meeting
 * becomes text is the Dictation page's choice; the page and the card's hint name it.
 */
import type { EntrySource, UserSettings } from "../../api";
import type { TabLifecycle } from "../settings";
import { mountRecent } from "../recent";
import { mountLanguageCard } from "./language-card";
import { mountMeetingCard } from "./meeting-card";
import { modelNoteHtml } from "./mode-rows";

/** Adds the Meetings page to the end of `container`; `windowHidden` keeps the
 *  card from polling until the window is shown, and All in History calls `openHistory`
 *  with the meetings filter. */
export function renderMeetings(
  container: HTMLElement,
  settings: UserSettings,
  windowHidden: boolean,
  openHistory: (source: EntrySource) => void,
): TabLifecycle {
  container.insertAdjacentHTML(
    "beforeend",
    `
    <h2 class="panel-title">Meetings</h2>
    <p class="panel-subtitle">Record a call in any app and get its text.</p>
    ${modelNoteHtml(settings.stt_mode)}
    <div id="meeting-card"></div>
    <div id="meetings-recent"></div>
  `,
  );

  const card = mountMeetingCard(
    container.querySelector<HTMLElement>("#meeting-card")!,
    settings,
    windowHidden,
  );
  const recent = mountRecent(
    container.querySelector<HTMLElement>("#meetings-recent")!,
    "meeting",
    openHistory,
  );
  const language = mountLanguageCard(container, settings, {
    idPrefix: "meetings",
    field: "meetings_language",
    detectHint: "For calls in several languages",
    listTitle: "Meeting language",
  });

  return {
    destroy: () => {
      language.destroy();
      card.destroy();
      recent.destroy();
    },
    releaseResources: () => {
      card.releaseResources?.();
      recent.releaseResources();
    },
    resumeResources: () => {
      card.resumeResources?.();
      recent.resumeResources();
    },
  };
}
