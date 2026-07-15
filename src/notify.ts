// Pure — no Tauri import, fully unit-testable without mocks.
export function onConnectivityChange(
  wasOffline: boolean,
  isOffline: boolean,
): { offline: boolean; shouldNotify: boolean } {
  return { offline: isOffline, shouldNotify: isOffline && !wasOffline };
}

// Side-effecting — lazily requests permission once, then sends a toast.
// Never throws; swallows and logs any failure so callers never need a
// try/catch of their own.
let permissionChecked = false;
export async function notifyError(body: string, title = "JustSay"): Promise<void> {
  try {
    const { isPermissionGranted, requestPermission, sendNotification } =
      await import("@tauri-apps/plugin-notification");
    if (!permissionChecked) {
      permissionChecked = true;
      if (!(await isPermissionGranted())) {
        await requestPermission();
      }
    }
    if (await isPermissionGranted()) {
      sendNotification({ title, body });
    }
  } catch (e) {
    console.warn("Toast notification failed:", e);
  }
}
