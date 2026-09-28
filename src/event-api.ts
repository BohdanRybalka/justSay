/** One import of the Tauri event API for every emit and subscription on a
 *  page; a failed import is tried again on the next call. */

let eventApi: Promise<typeof import("@tauri-apps/api/event")> | null = null;

export function loadEventApi(): Promise<typeof import("@tauri-apps/api/event")> {
  if (!eventApi) {
    eventApi = import("@tauri-apps/api/event").catch((e) => {
      eventApi = null;
      throw e;
    });
  }
  return eventApi;
}
