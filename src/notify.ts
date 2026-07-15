// Pure — no Tauri import, fully unit-testable without mocks.
export function onConnectivityChange(
  wasOffline: boolean,
  isOffline: boolean,
): { offline: boolean; shouldNotify: boolean } {
  return { offline: isOffline, shouldNotify: isOffline && !wasOffline };
}

export interface ConnectionCheckState {
  offline: boolean;
  firstCheckDone: boolean;
}

// Composes onConnectivityChange() with the cold-start firstCheckDone gate
// into one testable unit, so widget.ts's checkConnection() has nothing
// left to get wrong beyond wiring the health check and DOM update.
export function nextConnectionCheckState(
  current: ConnectionCheckState,
  healthOk: boolean,
): ConnectionCheckState & { shouldNotify: boolean } {
  const { offline, shouldNotify } = onConnectivityChange(current.offline, !healthOk);
  return {
    offline,
    firstCheckDone: true,
    shouldNotify: shouldNotify && current.firstCheckDone,
  };
}

// Side-effecting — lazily requests permission once, then sends a toast.
// Never throws; swallows and logs any failure so callers never need a
// try/catch of their own.
let permissionRequest: Promise<boolean> | null = null;

// Plain (non-async) function: the `if (!permissionRequest)` check-and-assign below
// executes as one uninterruptible synchronous step — no `await` sits between reading
// and writing `permissionRequest`, so two notifyError() calls racing in the same tick
// can never both see it unset. This is what a boolean flag couldn't guarantee, because
// the flag was written only after crossing an `await` boundary.
function ensureNotificationPermission(
  isPermissionGranted: () => Promise<boolean>,
  requestPermission: () => Promise<string>,
): Promise<boolean> {
  if (!permissionRequest) {
    permissionRequest = (async () => {
      if (await isPermissionGranted()) return true;
      const state = await requestPermission();
      return state === "granted";
    })().catch((e) => {
      // A *rejected* attempt is not a "decision" — only a resolved true/false is.
      // Reset so the next notifyError() call gets a fresh attempt instead of
      // permanently re-throwing this same dead promise for the rest of the session.
      permissionRequest = null;
      throw e;
    });
  }
  return permissionRequest;
}

export async function notifyError(body: string, title = "JustSay"): Promise<void> {
  try {
    const { isPermissionGranted, requestPermission, sendNotification } =
      await import("@tauri-apps/plugin-notification");
    if (await ensureNotificationPermission(isPermissionGranted, requestPermission)) {
      sendNotification({ title, body });
    }
  } catch (e) {
    console.warn("Toast notification failed:", e);
  }
}
