/**
 * The "Paste where I'm typing" row of Dictation's everyday card. While paste is
 * on and macOS has not let JustSay send keys, the hint says where to allow it
 * and a button opens that pane; the check repeats when the window regains
 * focus, which is how the user comes back from System Settings.
 */
import type { UserSettings } from "../../api";
import { saveSettings } from "../settings";
import { notifyError } from "../../notify";
import { renderToggle } from "../../ui/controls";

const PASTE_HINT = "Text lands at the cursor as soon as it's ready";
export const PERMISSION_HINT =
  "Allow JustSay in System Settings → Privacy & Security → Accessibility";
const RESULT_HINT = "setting-row-hint--result";

export const PASTE_ROW = `
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Paste where I'm typing</div>
          <div class="setting-row-hint" id="paste-hint">${PASTE_HINT}</div>
        </div>
        <div class="setting-row-controls">
          <button type="button" class="btn btn-small" id="btn-allow-paste" hidden>Open settings</button>
          <button id="paste-toggle" aria-label="Paste where I'm typing"></button>
        </div>
      </div>`;

async function invokeShell<T>(command: string): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command);
}

/** Wires `PASTE_ROW`, already inside `container`; `onSaved` runs after the
 *  switch is saved. Returns the row's teardown. */
export function wirePasteRow(
  container: HTMLElement,
  settings: UserSettings,
  onSaved: () => void,
): () => void {
  const toggle = container.querySelector<HTMLButtonElement>("#paste-toggle")!;
  const hint = container.querySelector<HTMLElement>("#paste-hint")!;
  const allow = container.querySelector<HTMLButtonElement>("#btn-allow-paste")!;
  let on = settings.paste_at_cursor;
  let permitted = true;
  let destroyed = false;

  function draw() {
    const blocked = on && !permitted;
    hint.textContent = blocked ? PERMISSION_HINT : PASTE_HINT;
    hint.classList.toggle(RESULT_HINT, blocked);
    allow.hidden = !blocked;
    toggle.setAttribute("aria-checked", String(on));
  }

  async function checkPermission() {
    try {
      permitted = await invokeShell<boolean>("paste_permission_granted");
    } catch {
      permitted = true;
    }
    if (!destroyed) draw();
  }

  renderToggle(toggle, on, async (next) => {
    toggle.disabled = true;
    try {
      await saveSettings({ paste_at_cursor: next });
      on = next;
      onSaved();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
    }
    if (destroyed) return;
    toggle.disabled = false;
    draw();
  });

  allow.addEventListener("click", () => {
    invokeShell("open_accessibility_settings").catch((e) =>
      notifyError(e instanceof Error ? e.message : String(e)),
    );
  });

  const recheck = () => void checkPermission();
  window.addEventListener("focus", recheck);
  void checkPermission();

  return () => {
    destroyed = true;
    window.removeEventListener("focus", recheck);
  };
}
