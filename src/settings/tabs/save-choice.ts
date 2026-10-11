/** Saving a choice made on a page, and telling the other windows about it. */
import type { UserSettings } from "../../api";
import { notifyError } from "../../notify";
import { saveSettings } from "../settings";
import { emitSettingsChanged } from "./dictation";

/** Resolves `false`, after telling the user why, when the choice could not be saved. */
export async function saveChoice(updates: Partial<UserSettings>): Promise<boolean> {
  try {
    await saveSettings(updates);
  } catch (e) {
    notifyError(e instanceof Error ? e.message : String(e));
    return false;
  }
  void emitSettingsChanged();
  return true;
}
