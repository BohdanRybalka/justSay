/**
 * The latest meetings or files under a section's top card: running and failed jobs of that
 * kind first, then the newest finished entries, three in all, with a link into History.
 */
import { api, type EntrySource, type HistoryEntry } from "../api";
import { icon } from "../ui/icons";
import { createEntryCards } from "./history-entry";
import { createJobCards } from "./history-jobs";
import type { TabLifecycle } from "./settings";

export type RecentKind = Exclude<EntrySource, "dictation">;

export const RECENT_POLL_MS = 5000;
export const RECENT_SHOWN = 3;

const WORDS: Record<RecentKind, { heading: string; empty: string }> = {
  meeting: { heading: "RECENT MEETINGS", empty: "No meetings yet." },
  file: { heading: "RECENT FILES", empty: "No files yet." },
};

export interface Recent extends Required<TabLifecycle> {
  /** A job was just queued elsewhere in the window; its card should appear now. */
  jobStarted(): void;
}

/** Adds the list to the end of `container` and keeps it current while the window is on screen. */
export function mountRecent(
  container: HTMLElement,
  kind: RecentKind,
  openHistory: (source: EntrySource) => void,
): Recent {
  const root = document.createElement("div");
  root.className = "recent";
  root.innerHTML = `
    <div class="group-head">
      <div class="group-label">${icon("clock")}${WORDS[kind].heading}</div>
      <button type="button" class="see-all">All in History${icon("chev", "small")}</button>
    </div>
    <div class="timeline"></div>
    <p class="history-empty" hidden>${WORDS[kind].empty}</p>
  `;
  container.append(root);
  const rows = root.querySelector<HTMLElement>(".timeline")!;
  const empty = root.querySelector<HTMLElement>(".history-empty")!;
  root.querySelector(".see-all")!.addEventListener("click", () => openHistory(kind));

  let entries: HistoryEntry[] = [];
  let entryRows: HTMLElement[] = [];
  let jobCards: readonly HTMLElement[] = [];
  let loaded = false;
  let readSucceeded = false;
  let latestRead = 0;
  let pollTimer: number | null = null;
  let destroyed = false;

  const cards = createEntryCards((card) => {
    entries = entries.filter((entry) => entry.id !== card.dataset.id);
    entryRows = entryRows.filter((row) => row !== card);
    paint();
    void load();
  });

  function paint(): void {
    const jobsShown = jobCards.slice(0, RECENT_SHOWN);
    const shown = [...jobsShown, ...entryRows.slice(0, RECENT_SHOWN - jobsShown.length)];
    rows.replaceChildren(...shown);
    rows.hidden = shown.length === 0;
    empty.hidden = !loaded || shown.length > 0;
  }

  function adopt(fresh: HistoryEntry[]): void {
    const unchanged =
      loaded && fresh.length === entries.length && fresh.every((entry, i) => entry.id === entries[i].id);
    loaded = true;
    if (unchanged) return;
    entries = fresh;
    entryRows = fresh.map((entry) => cards.create(entry));
    paint();
  }

  async function load(): Promise<void> {
    const token = ++latestRead;
    readSucceeded = false;
    try {
      const page = await api.getHistory(RECENT_SHOWN, null, { source: kind, starred: false });
      if (destroyed || token !== latestRead) return;
      adopt(page.entries);
      readSucceeded = true;
    } catch (error) {
      console.error(error);
    }
  }

  const jobs = createJobCards(
    (shownCards) => {
      jobCards = shownCards.filter((card) => card.dataset.kind === kind);
      paint();
    },
    load,
    () => readSucceeded,
  );

  function startPolling(): void {
    if (pollTimer === null) {
      pollTimer = window.setInterval(() => {
        void load();
        void jobs.refresh();
      }, RECENT_POLL_MS);
    }
  }

  function stopPolling(): void {
    if (pollTimer !== null) window.clearInterval(pollTimer);
    pollTimer = null;
  }

  void load();
  void jobs.refresh();
  startPolling();

  return {
    destroy() {
      destroyed = true;
      stopPolling();
      jobs.pause();
      cards.destroy();
    },
    releaseResources() {
      stopPolling();
      jobs.pause();
    },
    resumeResources() {
      void load();
      void jobs.refresh();
      startPolling();
    },
    jobStarted: () => void jobs.refresh(),
  };
}
