/**
 * The Files page: a drop area and Choose a file. A file dropped anywhere in the
 * window or picked from the ring is caught by `file-transcription.ts`; this page
 * only shows where to drop one and opens the picker.
 */
import { DROP_AREA_CLASS, DROP_HINT } from "../file-transcription";
import { icon } from "../../ui/icons";

/** Adds the Files page to the end of `container`; Choose a file calls `pick`. */
export function renderFiles(container: HTMLElement, pick: () => void): void {
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
}
