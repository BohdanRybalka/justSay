import { withTimeout } from "./timeout";

/** The budget on the clipboard command, bridge import included: neither step
 *  has a bound of its own (ADR 028). Matched to the widget's other shell
 *  commands, because it is the same two steps against the same transport. */
const CLIPBOARD_WRITE_TIMEOUT_MS = 3000;

/** The paste command is the clipboard write plus up to a second waiting for
 *  the user's modifier keys and a tenth holding the chord. */
const PASTE_TIMEOUT_MS = CLIPBOARD_WRITE_TIMEOUT_MS + 2000;

/** Where dictated text ended up: typed into the focused app, only on the
 *  clipboard, or nowhere. */
export type Delivery = "pasted" | "copied" | "failed";

async function invokeShell<T>(command: string, text: string, budgetMs: number): Promise<T> {
  return withTimeout(
    (async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      return invoke<T>(command, { text });
    })(),
    budgetMs,
  );
}

/** Puts `text` on this computer's clipboard, marked so Windows Cloud Clipboard
 *  and Apple Universal Clipboard never sync it to other devices.
 *  `navigator.clipboard` cannot set that mark, so every copy in the app comes
 *  here. Resolves `false` when the shell refused or did not answer; never
 *  rejects. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await invokeShell("write_clipboard_text", text, CLIPBOARD_WRITE_TIMEOUT_MS);
    return true;
  } catch (e) {
    console.warn("Copying to the clipboard failed:", e);
    return false;
  }
}

/** Dictated text: copied the same way, then pasted into the focused app when
 *  `paste` is on. Never rejects; a shell that did not answer is `failed`. */
export async function deliverDictation(text: string, paste: boolean): Promise<Delivery> {
  if (!paste) return (await copyToClipboard(text)) ? "copied" : "failed";
  try {
    return await invokeShell<Delivery>("paste_text", text, PASTE_TIMEOUT_MS);
  } catch (e) {
    console.warn("Pasting the dictation failed:", e);
    return "failed";
  }
}
