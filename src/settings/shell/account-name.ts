/**
 * The name the main window shows for the person using it: the OS account's
 * name, read once from the shell, and the initials its avatar carries.
 */
import { withTimeout } from "../../timeout";

/** The budget on the shell command, bridge import included: neither step has
 *  a bound of its own (ADR 028). */
const OS_NAME_READ_TIMEOUT_MS = 3000;

/** The OS account's full or login name; empty when the shell refused or did
 *  not answer. Never rejects. */
export async function readOsDisplayName(): Promise<string> {
  try {
    return await withTimeout(
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        return await invoke<string>("os_display_name");
      })(),
      OS_NAME_READ_TIMEOUT_MS,
    );
  } catch (e) {
    console.warn("Reading the account name failed:", e);
    return "";
  }
}

/** The first letter of each of the name's first two words, capitalised. */
export function initialsOf(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .slice(0, 2)
    .map((word) => Array.from(word)[0].toLocaleUpperCase())
    .join("");
}
