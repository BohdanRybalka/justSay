/**
 * The main window's sidebar: the account row, the four sections and the
 * status line pinned to its bottom. Which panel is open is decided by
 * `settings.ts`; this module draws it.
 */
import { initialsOf } from "./account-name";

export type PanelName = "insights" | "history" | "dictation" | "settings" | "account";

export type BackendState = "ready" | "starting" | "offline" | "unauthorized";

const BACKEND_STATE_TEXT: Record<BackendState, string> = {
  ready: "Ready",
  starting: "Starting…",
  offline: "Offline",
  unauthorized: "Can't reach JustSay",
};

/** Unreachable reads Starting… while the window still waits for its first
 *  answer and Offline after; a reachable backend that refused a request (401)
 *  reads as unreachable to the user. */
export function backendStateOf(backend: {
  reachable: boolean;
  refusedRequest: boolean;
  starting: boolean;
}): BackendState {
  if (!backend.reachable) return backend.starting ? "starting" : "offline";
  return backend.refusedRequest ? "unauthorized" : "ready";
}

/** `diagnosis` becomes the hover text, for the one state that has a cause to
 *  read back off a screenshot. */
export function renderSidebarStatus(status: HTMLElement, state: BackendState, diagnosis: string | null): void {
  status.className = `sidebar-status sidebar-status--${state}`;
  status.querySelector(".sidebar-status-text")!.textContent = BACKEND_STATE_TEXT[state];
  if (diagnosis) status.title = diagnosis;
  else status.removeAttribute("title");
}

export function renderPanelSelection(sidebar: HTMLElement, panel: PanelName): void {
  sidebar.querySelectorAll<HTMLElement>("[data-panel]").forEach((item) => {
    item.setAttribute("aria-current", String(item.dataset.panel === panel));
  });
}

export function renderAccountRow(row: HTMLElement, name: string): void {
  row.querySelector(".avatar")!.textContent = initialsOf(name);
  row.querySelector(".account-row-name")!.textContent = name || "Account";
}
