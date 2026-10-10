/**
 * The Files page: a drop area and Choose a file, then the latest files. A file dropped
 * anywhere in the window or picked from the ring is caught by `file-transcription.ts`; this
 * page shows where to drop one, opens the picker and lists what came of it.
 */
import type { EntrySource } from "../../api";
import { DROP_AREA_CLASS, DROP_HINT } from "../file-transcription";
import { mountRecent, type Recent } from "../recent";
import { icon } from "../../ui/icons";

/** Adds the Files page to the end of `container`; Choose a file calls `pick`, and All in
 *  History calls `openHistory` with the files filter. */
export function renderFiles(
  container: HTMLElement,
  pick: () => void,
  openHistory: (source: EntrySource) => void,
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
  return mountRecent(container, "file", openHistory);
}
