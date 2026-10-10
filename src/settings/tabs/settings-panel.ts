/**
 * The Settings panel's own card — Appearance, Storage and Version — then the
 * dictionary, the folded API keys and Delete all history under it. Both
 * deletions ask once, inline, before anything is removed.
 */
import { api, type UserSettings } from "../../api";
import { notifyError } from "../../notify";
import { renderFold, renderSegmented, type SegmentedOption } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { applyAppTheme, type ThemePreference } from "../../ui/theme";
import { getCloudKeyStatus, saveSettings, type TabLifecycle } from "../settings";
import { emitSettingsChanged } from "./dictation";
import { renderDictionary } from "./dictionary";
import { renderKeys } from "./keys";
import { renderVersionRow } from "./version-row";

const THEMES: readonly SegmentedOption<ThemePreference>[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
  { value: "system", label: "System" },
];

function describeFailure(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function button(label: string, className = "btn"): HTMLButtonElement {
  const created = document.createElement("button");
  created.type = "button";
  created.className = className;
  created.textContent = label;
  return created;
}

/** Puts Cancel and `confirmLabel` in place of what `controls` holds until one
 *  is pressed, then puts it back with focus on the button that asked.
 *  Resolves true only for `confirmLabel`. */
function confirmInline(controls: HTMLElement, confirmLabel: string): Promise<boolean> {
  const held = [...controls.childNodes];
  const asker = document.activeElement;
  const cancel = button("Cancel");
  const confirm = button(confirmLabel, "btn btn-primary");
  controls.replaceChildren(cancel, confirm);
  cancel.focus();
  return new Promise((resolve) => {
    const answer = (yes: boolean): void => {
      controls.replaceChildren(...held);
      if (asker instanceof HTMLElement && controls.contains(asker)) asker.focus();
      resolve(yes);
    };
    cancel.addEventListener("click", () => answer(false));
    confirm.addEventListener("click", () => answer(true));
  });
}

function renderAppearance(choice: HTMLElement, settings: UserSettings): void {
  let saved = settings.theme;
  const show = (theme: ThemePreference): void =>
    renderSegmented(choice, THEMES, theme, (next) => void choose(next));
  const choose = async (theme: ThemePreference): Promise<void> => {
    void applyAppTheme(theme);
    try {
      await saveSettings({ theme });
      saved = theme;
      await emitSettingsChanged();
    } catch (e) {
      show(saved);
      void applyAppTheme(saved);
      void notifyError(`Could not save the theme: ${describeFailure(e)}`);
    }
  };
  show(saved);
}

function renderStorageRow(row: HTMLElement, isDestroyed: () => boolean): void {
  row.innerHTML = `
    <div class="setting-row-text">
      <div class="setting-row-title">Storage</div>
      <div class="setting-row-hint storage-size">…</div>
    </div>
    <div class="setting-row-controls">
      <button type="button" class="btn storage-open">${icon("folder", "small")}Open folder</button>
      <button type="button" class="btn storage-clear">Clear</button>
    </div>
  `;
  const hint = row.querySelector<HTMLElement>(".storage-size")!;
  const controls = row.querySelector<HTMLElement>(".setting-row-controls")!;
  const clear = row.querySelector<HTMLButtonElement>(".storage-clear")!;
  let size: number | null = null;

  const showSize = (): void => {
    if (size === null) hint.textContent = "Size of temporary audio unknown";
    else hint.innerHTML = `<span class="num">${formatBytes(size)}</span> of temporary audio`;
  };
  const load = async (): Promise<void> => {
    try {
      const info = await api.getStorageInfo();
      if (isDestroyed()) return;
      size = info.temp_size_bytes;
    } catch {
      if (isDestroyed()) return;
      size = null;
    }
    showSize();
  };

  row.querySelector(".storage-open")!.addEventListener("click", async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("open_scratch_folder");
    } catch (e) {
      void notifyError(`Could not open the folder: ${describeFailure(e)}`);
    }
  });

  clear.addEventListener("click", async () => {
    hint.textContent =
      size === null ? "Clear the temporary audio?" : `Clear ${formatBytes(size)} of temporary audio?`;
    if (!(await confirmInline(controls, "Clear"))) {
      showSize();
      return;
    }
    clear.disabled = true;
    hint.textContent = "Clearing…";
    try {
      await api.cleanupTemp();
    } catch (e) {
      if (isDestroyed()) return;
      clear.disabled = false;
      hint.textContent = `Could not clear: ${describeFailure(e)}`;
      return;
    }
    if (isDestroyed()) return;
    clear.disabled = false;
    await load();
  });

  void load();
}

function renderDeleteHistory(area: HTMLElement, isDestroyed: () => boolean): void {
  area.innerHTML = `
    <div class="history-delete-status"></div>
    <div class="history-delete-controls">
      <button type="button" class="btn history-delete-button">Delete all history</button>
    </div>
  `;
  const status = area.querySelector<HTMLElement>(".history-delete-status")!;
  const controls = area.querySelector<HTMLElement>(".history-delete-controls")!;
  const start = area.querySelector<HTMLButtonElement>(".history-delete-button")!;

  start.addEventListener("click", async () => {
    status.textContent = "Delete every recording? This can't be undone.";
    if (!(await confirmInline(controls, "Delete"))) {
      status.textContent = "";
      return;
    }
    start.disabled = true;
    status.textContent = "Deleting…";
    try {
      await api.clearHistory();
      if (isDestroyed()) return;
      status.textContent = "History deleted";
    } catch (e) {
      if (isDestroyed()) return;
      status.textContent = `Could not delete: ${describeFailure(e)}`;
    }
    start.disabled = false;
  });
}

export function renderSettingsPanel(container: HTMLElement, settings: UserSettings): TabLifecycle {
  let destroyed = false;
  const isDestroyed = (): boolean => destroyed;
  container.insertAdjacentHTML(
    "beforeend",
    `
    <h2 class="panel-title">Settings</h2>
    <p class="panel-subtitle">Set once, forget about it.</p>
    <div class="card">
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Appearance</div>
          <div class="setting-row-hint">Follows your system by default</div>
        </div>
        <div class="setting-row-controls"><div class="theme-choice" aria-label="Appearance"></div></div>
      </div>
      <div class="setting-row storage-row"></div>
      <div class="setting-row version-row"></div>
    </div>
    <div class="dictionary"></div>
    <details class="api-keys"><summary>API keys</summary><div class="api-keys-rows"></div></details>
    <div class="history-delete"></div>
  `,
  );
  renderAppearance(container.querySelector<HTMLElement>(".theme-choice")!, settings);
  renderStorageRow(container.querySelector<HTMLElement>(".storage-row")!, isDestroyed);
  renderVersionRow(container.querySelector<HTMLElement>(".version-row")!, isDestroyed);
  const dictionary = renderDictionary(container.querySelector<HTMLElement>(".dictionary")!, settings);
  renderFold(container.querySelector<HTMLDetailsElement>(".api-keys")!);
  renderKeys(container.querySelector<HTMLElement>(".api-keys-rows")!, settings, getCloudKeyStatus());
  renderDeleteHistory(container.querySelector<HTMLElement>(".history-delete")!, isDestroyed);
  return {
    destroy: () => {
      destroyed = true;
      dictionary.destroy();
    },
  };
}
