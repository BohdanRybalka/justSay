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

let permissionRequest: Promise<boolean> | null = null;

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
