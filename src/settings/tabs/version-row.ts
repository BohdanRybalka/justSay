/**
 * The Version row of the Settings card: the running version, when updates were
 * last found to be none, and the updater's check, install and restart. The install
 * runs in the shell, which refuses it while the backend reports a recording or a
 * transcription in progress, asking before the download and again right before
 * installing; on Windows it closes JustSay the way Quit does before the installer
 * starts. The time of the last "up to date" answer is UI state kept in `localStorage`.
 */
import { setDisabled } from "../../ui/controls";
import { icon } from "../../ui/icons";

const LAST_UP_TO_DATE_KEY = "justsay.updates.lastUpToDateAt";

type Work = "meeting" | "dictation" | "transcription";

export const UPDATE_WAITS_FOR: Readonly<Record<Work, string>> = {
  meeting: "Stop the meeting recording first, then install the update.",
  dictation: "Finish the dictation first, then install the update.",
  transcription: "Wait until the transcription finishes, then install the update.",
};

function describeFailure(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** The two recognised shapes both mean "the release exists but its manifest is
 *  not usable yet", which reads as a broken app unless it is named. */
function describeUpdateCheckFailure(err: unknown): string {
  const raw = describeFailure(err);
  const lower = raw.toLowerCase();
  if (
    lower.includes("did not respond with a successful status code") ||
    lower.includes("could not fetch a valid release json") ||
    lower.includes("couldn't fetch a valid release json") ||
    lower.includes("couldnt fetch a valid release json")
  ) {
    return (
      "Check failed: the release manifest is not published yet. " +
      "Make sure the latest GitHub release is no longer marked as Draft."
    );
  }
  if (lower.includes("signature") || lower.includes("pubkey")) {
    return (
      "Check failed: the release manifest is not signed with the key this " +
      "build trusts. Re-run the release workflow with TAURI_SIGNING_PRIVATE_KEY set."
    );
  }
  return `Check failed: ${raw}`;
}

/** What the status line says when `install_update` rejects, and whether the button then
 *  offers a retry: the shell refuses with `{waits_for}`, `{could_not_check}` or `{failed}`. */
function describeInstallRefusal(err: unknown): { text: string; retry: boolean } {
  if (typeof err === "object" && err !== null) {
    if ("waits_for" in err) return { text: UPDATE_WAITS_FOR[err.waits_for as Work], retry: false };
    if ("could_not_check" in err) {
      return {
        text: `Couldn't check whether a recording is running, so the update waits: ${String(err.could_not_check)}`,
        retry: false,
      };
    }
    if ("failed" in err) return { text: `Install failed: ${String(err.failed)}`, retry: true };
  }
  return { text: `Install failed: ${describeFailure(err)}`, retry: true };
}

function readLastUpToDate(): number | null {
  try {
    const stored = Number(localStorage.getItem(LAST_UP_TO_DATE_KEY));
    return Number.isFinite(stored) && stored > 0 ? stored : null;
  } catch {
    return null;
  }
}

function rememberUpToDate(at: number | null): void {
  try {
    if (at === null) localStorage.removeItem(LAST_UP_TO_DATE_KEY);
    else localStorage.setItem(LAST_UP_TO_DATE_KEY, String(at));
  } catch {}
}

function timeSince(then: number, now: number): string {
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 60) return "just now";
  const relative = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (seconds < 3600) return relative.format(-Math.floor(seconds / 60), "minute");
  if (seconds < 86400) return relative.format(-Math.floor(seconds / 3600), "hour");
  return relative.format(-Math.floor(seconds / 86400), "day");
}

function upToDateText(at: number): string {
  return `Checked ${timeSince(at, Date.now())} · up to date`;
}

export function renderVersionRow(row: HTMLElement, isDestroyed: () => boolean): void {
  row.innerHTML = `
    <div class="setting-row-text">
      <div class="setting-row-title">Version <span class="num version-number"></span></div>
      <div class="setting-row-hint version-status"></div>
    </div>
    <div class="setting-row-controls"><button type="button" class="btn version-action"></button></div>
  `;
  const number = row.querySelector<HTMLElement>(".version-number")!;
  const status = row.querySelector<HTMLElement>(".version-status")!;
  const action = row.querySelector<HTMLButtonElement>(".version-action")!;
  let installReady = false;

  const offerCheck = (): void => {
    installReady = false;
    action.className = "btn version-action";
    action.innerHTML = `${icon("refresh", "small")}Check`;
    setDisabled(false, action);
  };
  const offerInstall = (label: string): void => {
    installReady = true;
    action.className = "btn btn-primary version-action";
    action.textContent = label;
    setDisabled(false, action);
  };
  const busy = (label: string, text: string): void => {
    setDisabled(true, action);
    action.textContent = label;
    status.textContent = text;
  };

  async function check(): Promise<void> {
    busy("Checking…", "Checking for updates…");
    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const found = await check();
      if (isDestroyed()) return;
      if (!found) {
        const at = Date.now();
        rememberUpToDate(at);
        status.textContent = upToDateText(at);
        offerCheck();
        return;
      }
      rememberUpToDate(null);
      status.textContent = `Version ${found.version} is ready`;
      offerInstall("Install and restart");
    } catch (err) {
      if (isDestroyed()) return;
      status.textContent = describeUpdateCheckFailure(err);
      offerCheck();
    }
  }

  async function install(): Promise<void> {
    const label = action.textContent ?? "";
    busy("Installing…", "Downloading and installing update…");
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("install_update");
    } catch (err) {
      if (isDestroyed()) return;
      const refusal = describeInstallRefusal(err);
      status.textContent = refusal.text;
      offerInstall(refusal.retry ? "Try again" : label);
      return;
    }
    try {
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (err) {
      if (isDestroyed()) return;
      status.textContent =
        `The update is installed. Restarting JustSay failed: ${describeFailure(err)}. ` +
        "Close and reopen the app to finish.";
      offerCheck();
    }
  }

  action.addEventListener("click", () => {
    if (action.disabled) return;
    void (installReady ? install() : check());
  });

  const lastUpToDate = readLastUpToDate();
  status.textContent = lastUpToDate === null ? "Never checked for updates" : upToDateText(lastUpToDate);
  offerCheck();

  void (async () => {
    try {
      const { getVersion } = await import("@tauri-apps/api/app");
      const version = await getVersion();
      if (!isDestroyed()) number.textContent = version;
    } catch {
      if (!isDestroyed()) number.textContent = "unknown";
    }
  })();
}
