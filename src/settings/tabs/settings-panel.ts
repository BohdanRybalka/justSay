/**
 * The Settings panel's own card, drawn above the old General content the panel
 * still hosts. Appearance picks the theme every window uses, saves it, and
 * tells the other windows to re-read it.
 */
import type { UserSettings } from "../../api";
import { notifyError } from "../../notify";
import { renderSegmented, type SegmentedOption } from "../../ui/controls";
import { applyAppTheme, type ThemePreference } from "../../ui/theme";
import { saveSettings } from "../settings";
import { emitSettingsChanged } from "./general";

const THEMES: readonly SegmentedOption<ThemePreference>[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
  { value: "system", label: "System" },
];

export function renderSettingsPanel(container: HTMLElement, settings: UserSettings): void {
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
    </div>
  `,
  );
  const choice = container.querySelector<HTMLElement>(".theme-choice")!;
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
      void notifyError(`Could not save the theme: ${e instanceof Error ? e.message : String(e)}`);
    }
  };
  show(saved);
}
