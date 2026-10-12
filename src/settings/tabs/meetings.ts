/**
 * The Meetings page: the card that starts and stops a meeting, the latest meetings, then LANGUAGE
 * (how a meeting is heard: detected for each part, or one picked from the
 * list) and MODEL (where a stopped meeting becomes text, apart from dictation;
 * the card's hint names it).
 */
import type { EntrySource, UserSettings } from "../../api";
import type { TabLifecycle } from "../settings";
import { mountRecent } from "../recent";
import { mountEngineCard } from "./engine-card";
import { mountLanguageCard } from "./language-card";
import { mountMeetingCard } from "./meeting-card";

const CLOUD_HINT = "Everyone's voices go to your cloud provider";

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
  const engine = mountEngineCard(
    container,
    settings,
    {
      idPrefix: "meetings",
      field: "meetings_engine",
      ariaLabel: "Meeting model",
      cloudHint: CLOUD_HINT,
      onShown: (shown) => card.setEngine(shown),
    },
    windowHidden,
  );

  return {
    destroy: () => {
      language.destroy();
      engine.destroy();
      card.destroy();
      recent.destroy();
    },
    releaseResources: () => {
      card.releaseResources?.();
      recent.releaseResources();
      engine.releaseResources();
    },
    resumeResources: () => {
      card.resumeResources?.();
      recent.resumeResources();
      engine.resumeResources();
    },
  };
}
