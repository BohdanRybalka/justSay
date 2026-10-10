import {
  api,
  EVERY_ENTRY,
  MalformedResponseError,
  SidecarTooOldError,
  type EntrySource,
  type HistoryEntry,
  type HistoryFilter,
  type UserSettings,
} from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import { renderSegmented, type SegmentedOption } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { createEntryCards, EMPTY_KIND, SOURCE_ICONS } from "../history-entry";
import {
  createHistoryList,
  sidecarTooOldText,
  SENTINEL_READING,
  type BuiltRow,
  type HistoryRowsClaim,
} from "../history-list";
import { closeMatchesHeading, createTimelineRows, matchDayGroups } from "../history-timeline";
import { createJobCards } from "../history-jobs";
import { escapeHtml } from "../html";
import type { TabLifecycle } from "../settings";

const SEARCH_DEBOUNCE_MS = 300;
export const NEWER_POLL_MS = 5000;
const PAGE_SIZE = 30;

type KindChip = "all" | EntrySource;

const KIND_CHIPS: readonly SegmentedOption<KindChip>[] = [
  { value: "all", label: "All" },
  { value: "dictation", label: "Dictation", icon: SOURCE_ICONS.dictation },
  { value: "meeting", label: "Meetings", icon: SOURCE_ICONS.meeting },
  { value: "file", label: "Files", icon: SOURCE_ICONS.file },
];

const SKELETON_CARD = `<div class="entry entry--skeleton" aria-hidden="true"><span class="entry-dot"></span><i></i><i></i></div>`;

/** The panel's title and subtitle, which sit above whatever the panel hosts before the timeline. */
export function renderHistoryHeading(container: HTMLElement): void {
  container.insertAdjacentHTML(
    "beforeend",
    `<h2 class="panel-title">History</h2>
    <p class="panel-subtitle">Everything you've said, kept on this machine.</p>`,
  );
}

/** History as a day-grouped timeline under the search box and the kind and Starred chips,
 *  paging as it scrolls, picking up new recordings every few seconds while the window is on
 *  screen, and showing files and meetings still being transcribed at the top of today. */
export function renderHistory(
  container: HTMLElement,
  settings: UserSettings,
  source: EntrySource | null = null,
): TabLifecycle {
  const section = document.createElement("div");
  section.className = "history";
  section.innerHTML = `
    <div class="search-box">${icon("search")}<input type="search" id="history-search" placeholder="Search what you said" aria-label="Search" /></div>
    <div class="chips history-filter"><div id="history-kind" aria-label="Show"></div><span class="chips-sep"></span><button type="button" class="chip-btn chip-btn--star" id="history-starred" aria-pressed="false">${icon("star")}Starred</button></div>
    <div class="history-search-hint" id="history-search-hint"></div>
    <div class="history-count" id="history-count">Loading...</div>
    <div class="timeline">
      <div id="history-days"></div>
      <div class="timeline-more ${SENTINEL_READING}" id="history-more">${SKELETON_CARD.repeat(3)}</div>
    </div>
  `;
  container.append(section);

  const searchInput = section.querySelector<HTMLInputElement>("#history-search")!;
  const searchHint = section.querySelector<HTMLElement>("#history-search-hint")!;
  const daysEl = section.querySelector<HTMLElement>("#history-days")!;
  const timeline = createTimelineRows(daysEl);
  const entryCards = createEntryCards((card) => {
    timeline.rowRemoved(card);
    list.entryRemoved();
  });
  const shortcut = formatAccelerator(settings.shortcut, detectShortcutPlatform(navigator));

  let searchClaim: HistoryRowsClaim | null = null;
  let debounceTimer: number | null = null;
  let pollTimer: number | null = null;
  let destroyed = false;
  let filter: HistoryFilter = { ...EVERY_ENTRY, source };
  let jobCards: readonly HTMLElement[] = [];

  const list = createHistoryList({
    pageSize: PAGE_SIZE,
    noun: { singular: "recording", plural: "recordings" },
    featureName: "History",
    elements: {
      count: section.querySelector<HTMLElement>("#history-count")!,
      sentinel: section.querySelector<HTMLElement>("#history-more")!,
    },
    rows: timeline,
    createRow: entryCards.create,
    renderEmptyState: (isEmpty) => {
      if (!isEmpty) return;
      const named = filter.starred
        ? "Nothing starred yet."
        : filter.source === null || filter.source === "dictation"
          ? null
          : EMPTY_KIND[filter.source];
      daysEl.insertAdjacentHTML(
        "beforeend",
        named
          ? `<p class="history-empty">${named}</p>`
          : `<p class="history-empty">Nothing here yet. Hold <b>${escapeHtml(shortcut)}</b> anywhere and talk.</p>`,
      );
    },
    isDestroyed: () => destroyed,
    filter: () => filter,
  });

  function showCurrentView(): void {
    const query = searchInput.value.trim();
    if (query) void runSearch(query);
    else void list.load();
  }

  function showJobCards(): void {
    timeline.setPending(
      jobCards.filter((card) => filter.source === null || card.dataset.kind === filter.source),
    );
  }

  function applyFilter(next: HistoryFilter): void {
    filter = next;
    showJobCards();
    showCurrentView();
  }

  renderSegmented<KindChip>(
    section.querySelector<HTMLElement>("#history-kind")!,
    KIND_CHIPS,
    source ?? "all",
    (kind) => applyFilter({ ...filter, source: kind === "all" ? null : kind }),
    "chips",
  );

  const starredChip = section.querySelector<HTMLButtonElement>("#history-starred")!;
  starredChip.addEventListener("click", () => {
    starredChip.setAttribute("aria-pressed", String(!filter.starred));
    applyFilter({ ...filter, starred: !filter.starred });
  });

  function noMatchesElement(): HTMLElement {
    const el = document.createElement("p");
    el.className = "history-empty";
    el.textContent = "Nothing you said matches that.";
    return el;
  }

  /** Exact matches by day; under "Close matches", word parts by day and then, last, the
   *  entries found by meaning alone. */
  function searchTiers(rows: readonly BuiltRow[]): HTMLElement[] {
    const kind = (match: HistoryEntry["match"]) => rows.filter((row) => row.entry.match === match);
    const near = matchDayGroups(kind("near"));
    const meaning = matchDayGroups(
      rows.filter((row) => row.entry.match !== "exact" && row.entry.match !== "near"),
    );
    const painted = matchDayGroups(kind("exact"));
    if (near.length + meaning.length > 0) painted.push(closeMatchesHeading(), ...near, ...meaning);
    return painted;
  }

  /**
   * The search lane, holding a claim on the shared rows for as long as what is on
   * screen is its own paint.
   *
   * The claim is taken before the request, so a page load already in flight is
   * superseded the moment the user asks for matches. The hint belongs to this
   * tab, not the list: a superseded search clears the hint it put up only while
   * no newer search owns it, so a live error message is never blanked. A
   * `MalformedResponseError` names an endpoint, which is for the log, not the user.
   */
  async function runSearch(q: string) {
    const claim = list.claimRows();
    searchClaim = claim;
    searchHint.textContent = "Searching...";
    try {
      const resp = await api.searchHistory(q, PAGE_SIZE, filter);
      const built = resp.entries.map((entry) => ({ entry, element: entryCards.create(entry) }));
      claim.replaceRows(built.length === 0 ? [noMatchesElement()] : searchTiers(built));
      claim.renderCount(`${resp.total} match${resp.total !== 1 ? "es" : ""}`);
      claim.renderMore(false);
      if (claim.isCurrent()) {
        searchHint.textContent = "";
      }
    } catch (e) {
      if (!claim.isCurrent()) return;
      if (e instanceof SidecarTooOldError) {
        searchHint.textContent = sidecarTooOldText("Search");
      } else if (e instanceof MalformedResponseError) {
        searchHint.textContent = "Search failed";
        console.error(e);
      } else {
        const msg = (e as Error).message || "Search failed";
        searchHint.textContent = msg.toLowerCase().includes("invalid")
          ? "Invalid search query"
          : msg;
      }
    } finally {
      claim.release();
      if (!destroyed && searchClaim === claim && !claim.isCurrent()) {
        searchHint.textContent = "";
      }
    }
  }

  searchInput.addEventListener("input", () => {
    if (debounceTimer !== null) {
      window.clearTimeout(debounceTimer);
    }
    const value = searchInput.value.trim();
    debounceTimer = window.setTimeout(() => {
      debounceTimer = null;
      if (!value) searchHint.textContent = "";
      showCurrentView();
    }, SEARCH_DEBOUNCE_MS);
  });

  const jobs = createJobCards(
    (cards) => {
      jobCards = cards;
      showJobCards();
    },
    () => list.loadNewer(),
    (entryId) =>
      filter.source !== null ||
      filter.starred ||
      searchInput.value.trim() !== "" ||
      Array.from(daysEl.querySelectorAll<HTMLElement>(".entry")).some((el) => el.dataset.id === entryId),
  );

  function startPolling(): void {
    if (pollTimer === null) {
      pollTimer = window.setInterval(() => {
        void list.loadNewer();
        void jobs.refresh();
      }, NEWER_POLL_MS);
    }
  }

  function stopPolling(): void {
    if (pollTimer !== null) window.clearInterval(pollTimer);
    pollTimer = null;
  }

  void list.load();
  void jobs.refresh();
  startPolling();

  return {
    destroy() {
      destroyed = true;
      stopPolling();
      jobs.pause();
      list.disconnect();
      entryCards.destroy();
      if (debounceTimer !== null) window.clearTimeout(debounceTimer);
    },
    releaseResources() {
      stopPolling();
      jobs.pause();
    },
    resumeResources() {
      void list.loadNewer();
      void jobs.refresh();
      startPolling();
    },
  };
}
