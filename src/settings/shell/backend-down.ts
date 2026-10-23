/**
 * What the main window can do while the backend is not running: read why, as
 * the shell recorded it, and restart JustSay.
 */
import { withTimeout } from "../../timeout";

/** The budget on the shell command, bridge import included: neither step has
 *  a bound of its own (ADR 028). */
const FAILURE_READ_TIMEOUT_MS = 3000;

/** The reason the shell recorded, or `null` when it has none or did not
 *  answer. Never rejects. */
export async function readBackendFailure(): Promise<string | null> {
  try {
    return await withTimeout(
      (async () => {
        const { invoke } = await import("@tauri-apps/api/core");
        return await invoke<string | null>("backend_failure_reason");
      })(),
      FAILURE_READ_TIMEOUT_MS,
    );
  } catch (e) {
    console.warn("Reading why the backend is down failed:", e);
    return null;
  }
}

/** Quit JustSay and start it again, which starts a new backend. */
export async function restartJustSay(): Promise<void> {
  const { relaunch } = await import("@tauri-apps/plugin-process");
  await relaunch();
}
