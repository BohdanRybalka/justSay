/**
 * MODEL on a page that picks its own engine (Meetings, Files): the Cloud and Local model rows,
 * saved on press. A press while the last one is still saving is ignored, and a choice that could
 * not be saved is shown as it was.
 */
import type { UserSettings } from "../../api";
import { icon } from "../../ui/icons";
import { getCloudKeyStatus } from "../settings";
import { CLOUD_KEY_MISSING, cloudKeyMissing, drawRow, modeRowHtml } from "./mode-rows";
import { saveChoice } from "./save-choice";

type Engine = UserSettings["files_engine"];

const LOCAL_HINT = "Stays on this computer";

export interface EngineCardSpec {
  /** Prefix of the row ids: `<prefix>-cloud`, `<prefix>-local`. */
  idPrefix: string;
  field: "meetings_engine" | "files_engine";
  ariaLabel: string;
  cloudHint: string;
  /** Called with the engine every time the rows are drawn, so a page can say where it runs. */
  onShown?: (engine: Engine) => void;
}

/** Adds the MODEL group to the end of `container`. `destroy` keeps a save still in flight from
 *  drawing on a page that is gone. */
export function mountEngineCard(
  container: HTMLElement,
  settings: UserSettings,
  spec: EngineCardSpec,
): { destroy: () => void } {
  const { idPrefix, field } = spec;
  let engine: Engine = settings[field];
  let saving = false;
  let destroyed = false;

  container.insertAdjacentHTML(
    "beforeend",
    `
    <div class="group-label">${icon("chip")}MODEL</div>
    <div class="card" role="radiogroup" aria-label="${spec.ariaLabel}">
      ${modeRowHtml(`${idPrefix}-cloud`, "cloud", "Cloud")}
      ${modeRowHtml(`${idPrefix}-local`, "chip", "Local model")}
    </div>
  `,
  );

  const cloudRow = container.querySelector<HTMLButtonElement>(`#${idPrefix}-cloud`)!;
  const localRow = container.querySelector<HTMLButtonElement>(`#${idPrefix}-local`)!;
  const keyMissing = cloudKeyMissing(getCloudKeyStatus());

  function show(shown: Engine) {
    spec.onShown?.(shown);
    drawRow(cloudRow, shown === "cloud", {
      hint: keyMissing ? CLOUD_KEY_MISSING : spec.cloudHint,
      alert: keyMissing,
      locked: false,
      disabled: saving,
      action: "",
    });
    drawRow(localRow, shown === "local", {
      hint: LOCAL_HINT,
      alert: false,
      locked: false,
      disabled: saving,
      action: "",
    });
  }

  async function choose(next: Engine) {
    if (saving || next === engine) return;
    saving = true;
    show(next);
    const saved = await saveChoice({ [field]: next });
    if (destroyed) return;
    saving = false;
    if (saved) engine = next;
    show(engine);
  }

  cloudRow.addEventListener("click", () => void choose("cloud"));
  localRow.addEventListener("click", () => void choose("local"));
  show(engine);

  return {
    destroy: () => {
      destroyed = true;
    },
  };
}
