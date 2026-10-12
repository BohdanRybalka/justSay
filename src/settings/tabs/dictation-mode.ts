/**
 * MODEL on the Dictation page: the Cloud and Local model radio rows. Cloud says whether the
 * Groq key is set; Local draws the local engine's state while the window is shown, and its
 * Install downloads the model, starts the engine and then switches to Local. Picking an
 * installed row switches the mode.
 */
import { api, type LocalSTTStatus, type UserSettings } from "../../api";
import { getCloudKeyStatus, loadSettings, type TabLifecycle } from "../settings";
import { displayableError, notifyError } from "../../notify";
import { onIndicatorStateChange } from "../../status-indicator";
import { icon } from "../../ui/icons";
import { watchLocalModel } from "./local-model";
import {
  CLOUD_HINT,
  CLOUD_KEY_MISSING,
  cloudKeyMissing,
  drawRow,
  localModeRowHtml,
  localView,
  modeRowHtml,
} from "./mode-rows";

type Mode = UserSettings["stt_mode"];

let prevLastError: string | null = null;

/** Adds the MODEL group to the end of `container`. Nothing is
 *  read while `windowHidden`; the returned lifecycle stops and restarts the poll. */
export function renderDictationMode(
  container: HTMLElement,
  settings: UserSettings,
  windowHidden = false,
): TabLifecycle {
  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("chip")}MODEL</div>
    <div class="card" role="radiogroup" aria-label="Push-to-talk model">
      ${modeRowHtml("mode-cloud", "cloud", "Cloud")}
      ${localModeRowHtml("mode-local")}
    </div>
  `,
  );

  const cloudRow = container.querySelector<HTMLButtonElement>("#mode-cloud")!;
  const localRowEl = container.querySelector<HTMLButtonElement>("#mode-local")!;
  const keyMissing = cloudKeyMissing(getCloudKeyStatus());

  let currentMode: Mode = settings.stt_mode;

  function draw() {
    drawRow(cloudRow, currentMode === "cloud", {
      hint: keyMissing ? CLOUD_KEY_MISSING : CLOUD_HINT,
      alert: keyMissing,
      locked: false,
      disabled: false,
      action: "",
    });
    drawRow(localRowEl, currentMode === "local", localView(model.row(currentMode === "local")));
  }

  /** Only Local's reads count, so a failure already shown is not shown again
   *  after a trip through Cloud. */
  function reportNewFailure(status: LocalSTTStatus) {
    if (currentMode !== "local") return;
    const reported = status.last_error ?? null;
    if (onIndicatorStateChange(prevLastError, reported)) notifyError(displayableError(reported)!);
    prevLastError = reported;
  }

  const model = watchLocalModel({
    onChange: draw,
    onStatus: reportNewFailure,
    onInstalled: () => void switchMode("local"),
  });

  async function switchMode(mode: Mode) {
    if (currentMode === mode) return;
    if (mode === "cloud") model.declineSwitch();
    const previous = currentMode;
    currentMode = mode;
    draw();
    try {
      await api.setSttMode(mode);
    } catch (e) {
      currentMode = previous;
      draw();
      notifyError(e instanceof Error ? e.message : String(e));
      return;
    }
    await loadSettings().catch(() => {});
    void model.refresh();
  }

  async function tryAgain() {
    if (currentMode === "local" && model.read() !== "failed") {
      try {
        await api.sttLocalPrewarm();
      } catch {}
    }
    await model.refresh();
  }

  cloudRow.addEventListener("click", () => void switchMode("cloud"));
  localRowEl.addEventListener("click", () =>
    model.press(currentMode === "local", {
      choose: () => void switchMode("local"),
      retry: () => void tryAgain(),
    }),
  );

  draw();
  if (!windowHidden) model.resume();

  return {
    destroy: () => {
      model.release();
      prevLastError = null;
    },
    releaseResources: model.release,
    resumeResources: model.resume,
  } satisfies TabLifecycle;
}
