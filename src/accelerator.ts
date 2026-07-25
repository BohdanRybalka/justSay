export const DEFAULT_SHORTCUT = "Ctrl+Alt+KeyV";

export type ShortcutPlatform = "mac" | "windows";

type ModifierToken = "Ctrl" | "Alt" | "Shift" | "Super";

const MAC_DISPLAY_ORDER: readonly ModifierToken[] = ["Ctrl", "Alt", "Shift", "Super"];

const MAC_GLYPHS: Record<ModifierToken, string> = {
  Ctrl: "⌃",
  Alt: "⌥",
  Shift: "⇧",
  Super: "⌘",
};

const WINDOWS_LABELS: Record<ModifierToken, string> = {
  Ctrl: "Ctrl",
  Alt: "Alt",
  Shift: "Shift",
  Super: "Win",
};

const MODIFIER_TOKENS: ReadonlySet<string> = new Set(MAC_DISPLAY_ORDER);

const MODIFIER_KEY_NAMES: ReadonlySet<string> = new Set(["Control", "Shift", "Alt", "Meta"]);

const LEADING_CODE_PREFIX = /^(?:Key|Digit)/;

function isModifierToken(token: string): token is ModifierToken {
  return MODIFIER_TOKENS.has(token);
}

function splitAccelerator(accelerator: string): { modifiers: ModifierToken[]; key: string } {
  const modifiers: ModifierToken[] = [];
  const remainder: string[] = [];
  for (const token of accelerator.split("+")) {
    if (token.length === 0) continue;
    if (isModifierToken(token)) modifiers.push(token);
    else remainder.push(token);
  }
  return { modifiers, key: remainder.join("+") };
}

/**
 * Picks the rendering platform from a `navigator`-shaped source.
 *
 * The caller passes the real `navigator` so both branches stay reachable from a
 * test. `platform` is deprecated but present in WebView2 and WKWebView alike;
 * `userAgent` is the fallback for the day it stops reporting.
 */
export function detectShortcutPlatform(source: { platform?: string; userAgent?: string }): ShortcutPlatform {
  if (source.platform && /^Mac/i.test(source.platform)) return "mac";
  if (source.userAgent && /Mac OS X|Macintosh/.test(source.userAgent)) return "mac";
  return "windows";
}

/**
 * Builds the stored accelerator from a key press.
 *
 * The emitted string is platform-neutral — only `Super`, `Ctrl`, `Alt`, `Shift`
 * and one `KeyboardEvent.code`, in that order — because the `global-hotkey`
 * parser reads `Super` as Command on macOS and as the Windows key elsewhere
 * (`docs/adr/034-shortcut-storage-stays-platform-neutral.md`).
 *
 * `"pending"` means a modifier alone was pressed and the capture should keep
 * waiting; `"no-modifier"` means a bare key that cannot become a global hotkey.
 */
export function acceleratorFromKeyEvent(
  event: Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "altKey" | "shiftKey" | "metaKey">,
): { ok: true; accelerator: string } | { ok: false; reason: "pending" | "no-modifier" } {
  if (MODIFIER_KEY_NAMES.has(event.key)) return { ok: false, reason: "pending" };

  const held: ModifierToken[] = [];
  if (event.metaKey) held.push("Super");
  if (event.ctrlKey) held.push("Ctrl");
  if (event.altKey) held.push("Alt");
  if (event.shiftKey) held.push("Shift");

  if (held.length === 0) return { ok: false, reason: "no-modifier" };

  return { ok: true, accelerator: [...held, event.code].join("+") };
}

/**
 * Renders a stored accelerator for a human.
 *
 * macOS gets Apple's glyphs in Apple's canonical Control-Option-Shift-Command
 * order; Windows keeps the stored order and renames `Super` to `Win`. Only a
 * leading `Key` or `Digit` is stripped from the key code, so `ArrowUp`,
 * `Backquote` and `F5` survive intact.
 */
export function formatAccelerator(accelerator: string, platform: ShortcutPlatform): string {
  const { modifiers, key } = splitAccelerator(accelerator);
  const keyLabel = key.replace(LEADING_CODE_PREFIX, "");

  if (platform === "mac") {
    const glyphs = MAC_DISPLAY_ORDER.filter((token) => modifiers.includes(token)).map(
      (token) => MAC_GLYPHS[token],
    );
    return [...glyphs, keyLabel].join("");
  }

  return [...modifiers.map((token) => WINDOWS_LABELS[token]), keyLabel]
    .filter((part) => part.length > 0)
    .join(" + ");
}

/** The hint shown when a captured key press carried no modifier. */
export function modifierHint(platform: ShortcutPlatform): string {
  return platform === "mac"
    ? "Must include at least one modifier (⌘, ⌥, ⌃, ⇧)"
    : "Must include at least one modifier (Ctrl, Alt, Shift, Win)";
}

/**
 * Decides whether the widget must re-register the hotkey.
 *
 * Equality with the active accelerator is not enough: after a restart the string
 * can match while nothing is registered, which is exactly how a saved shortcut
 * ends up dead and a re-save of the same combination repairs nothing.
 */
export function shouldReapplyShortcut(
  desired: string,
  active: string | null,
  isCurrentlyRegistered: boolean,
): boolean {
  return desired !== active || !isCurrentlyRegistered;
}

/**
 * The user-facing text for a refused registration.
 *
 * The reason is the message the rejected `register` call returned, not a guess:
 * a fixed sentence would tell the user the same thing whatever the system said.
 */
export function shortcutFailureMessage(
  accelerator: string,
  reason: string,
  stillActive: string | null,
): string {
  const given = reason.trim().length > 0 ? reason.trim() : "no reason given";
  const stated = /[.!?]$/.test(given) ? given : `${given}.`;
  const refused = `Push-to-talk shortcut ${accelerator} could not be registered: ${stated}`;
  return stillActive
    ? `${refused} ${stillActive} is still active.`
    : `${refused} Push-to-talk is off until you choose another combination.`;
}
