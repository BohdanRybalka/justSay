import { withTimeout } from "./timeout";

/** The budget on the clipboard command, bridge import included: neither step
 *  has a bound of its own (ADR 028). Matched to the widget's other shell
 *  commands, because it is the same two steps against the same transport. */
const CLIPBOARD_WRITE_TIMEOUT_MS = 3000;

/** Puts `text` on this computer's clipboard, marked so Windows Cloud Clipboard
 *  and Apple Universal Clipboard never sync it to other devices.
 *  `navigator.clipboard` cannot set that mark, so every copy in the app comes
 *  here. Resolves `false` when the shell refused or did not answer; never
 *  rejects. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await withTimeout(
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("write_clipboard_text", { text });
      })(),
      CLIPBOARD_WRITE_TIMEOUT_MS,
    );
    return true;
  } catch (e) {
    console.warn("Copying to the clipboard failed:", e);
    return false;
  }
}
