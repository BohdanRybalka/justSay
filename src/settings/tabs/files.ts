/**
 * The Files page: a drop area and Choose a file, the latest files, then LANGUAGE and MODEL for
 * files alone. A file dropped anywhere in the window or picked from the ring is caught by
 * `file-transcription.ts`; this page shows where to drop one, opens the picker, lists what came
 * of it and says in which language and where a file becomes text.
 */
import type { EntrySource, UserSettings } from "../../api";
import { DROP_AREA_CLASS, DROP_HINT } from "../file-transcription";
import { mountRecent, type Recent } from "../recent";
import { icon } from "../../ui/icons";
import { mountEngineCard } from "./engine-card";
import { mountLanguageCard } from "./language-card";
import { CLOUD_HINT } from "./mode-rows";

/** Adds the Files page to the end of `container`; Choose a file calls `pick`, and All in
 *  History calls `openHistory` with the files filter. `windowHidden` keeps the model row from
 *  reading until the window is shown. */
export function renderFiles(
  container: HTMLElement,
  settings: UserSettings,
  pick: () => void,
  openHistory: (source: EntrySource) => void,
  windowHidden = false,
): Recent {
  container.insertAdjacentHTML(
    "beforeend",
    `
    <h2 class="panel-title">Files</h2>
    <p class="panel-subtitle">Turn a recording into text.</p>
    <div class="${DROP_AREA_CLASS}">
      <span class="brand-mark">${icon("upload", "large")}</span>
      <b class="drop-area-title">Drop an audio or video file here</b>
      <span class="drop-area-hint">${DROP_HINT}</span>
      <button type="button" class="btn" id="files-choose">${icon("folder", "small")}Choose a file</button>
    </div>
  `,
  );

  container.querySelector<HTMLButtonElement>("#files-choose")!.addEventListener("click", pick);
  const recent = mountRecent(container, "file", openHistory);
  const language = mountLanguageCard(container, settings, {
    idPrefix: "files",
    field: "files_language",
    detectHint: "Each file is heard in its own language",
    listTitle: "File language",
  });
  const engine = mountEngineCard(
    container,
    settings,
    {
      idPrefix: "files",
      field: "files_engine",
      ariaLabel: "File model",
      cloudHint: CLOUD_HINT,
    },
    windowHidden,
  );

  return {
    ...recent,
    destroy: () => {
      language.destroy();
      engine.destroy();
      recent.destroy();
    },
    releaseResources: () => {
      recent.releaseResources();
      engine.releaseResources();
    },
    resumeResources: () => {
      recent.resumeResources();
      engine.resumeResources();
    },
  };
}
