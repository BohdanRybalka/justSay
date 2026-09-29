import {
  api,
  MalformedResponseError,
  SidecarTooOldError,
  type HistoryEntry,
  type UserSettings,
} from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import { copyToClipboard } from "../../clipboard";
import { icon, type IconName } from "../../ui/icons";
import { createHistoryList, sidecarTooOldText, SENTINEL_READING, type HistoryRowsClaim } from "../history-list";
import { countOf, createTimelineRows, formatClock, formatDuration } from "../history-timeline";
import { escapeHtml } from "../html";
import type { TabLifecycle } from "../settings";

const SEARCH_DEBOUNCE_MS = 300;
export const NEWER_POLL_MS = 5000;
const PAGE_SIZE = 30;
const COPIED_FLASH_MS = 1500;
const SHOW_MORE = "Show more";
const SHOW_LESS = "Show less";

const SOURCE_ICONS: Record<HistoryEntry["source"], IconName> = {
  dictation: "mic",
  file: "file",
  meeting: "users",
};

const SKELETON_CARD = `<div class="entry entry--skeleton" aria-hidden="true"><span class="entry-dot"></span><i></i><i></i></div>`;

/** The panel's title and subtitle, which sit above whatever the panel hosts before the timeline. */
export function renderHistoryHeading(container: HTMLElement): void {
  container.insertAdjacentHTML(
    "beforeend",
    `<h2 class="panel-title">History</h2>
    <p class="panel-subtitle">Everything you've said, kept on this machine.</p>`,
  );
}

/** Marks every collapsed card whose text is cut off, which is what shows its "Show more".
 *  Runs whenever the timeline's size changes: a page landing, a card expanding, a resize. */
function markLongTexts(records: readonly ResizeObserverEntry[]): void {
  for (const record of records) {
    for (const card of record.target.querySelectorAll<HTMLElement>(".entry:not(.entry--expanded)")) {
      const text = card.querySelector<HTMLElement>(".entry-text");
      if (text) card.classList.toggle("entry--long", text.scrollHeight > text.clientHeight + 1);
    }
  }
}

function metaSeparator(): string {
  return `<span class="entry-meta-sep">·</span>`;
}

function sourceBadge(entry: HistoryEntry): string {
  if (entry.source === "file" && entry.source_name) {
    return `<span class="entry-source">${icon("file", "small")}${escapeHtml(entry.source_name)}</span>`;
  }
  if (entry.source === "meeting") {
    return `<span class="entry-source">${icon("users", "small")}meeting</span>`;
  }
  return "";
}

function metaLine(entry: HistoryEntry): string {
  const parts = [entry.timestamp == null ? "—" : formatClock(new Date(entry.timestamp))];
  if (entry.audio_duration_seconds != null) {
    parts.push(`<span class="num">${formatDuration(entry.audio_duration_seconds)}</span>`);
  }
  if (entry.word_count != null) parts.push(countOf(entry.word_count, "word", "words"));
  return parts.map((part) => `<span>${part}</span>`).join(metaSeparator());
}

/** History as a day-grouped timeline under the old search box, paging as it scrolls and
 *  picking up new recordings every few seconds while the window is on screen. */
export function renderHistory(container: HTMLElement, settings: UserSettings): TabLifecycle {
  const section = document.createElement("div");
  section.className = "history";
  section.innerHTML = `
    <div class="search-box">${icon("search")}<input type="search" id="history-search" placeholder="Search transcripts..." aria-label="Search" /></div>
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
  const textFit = new ResizeObserver(markLongTexts);
  textFit.observe(daysEl);
  const shortcut = formatAccelerator(settings.shortcut, detectShortcutPlatform(navigator));

  let searchClaim: HistoryRowsClaim | null = null;
  let debounceTimer: number | null = null;
  let pollTimer: number | null = null;
  let destroyed = false;

  const list = createHistoryList({
    pageSize: PAGE_SIZE,
    noun: { singular: "recording", plural: "recordings" },
    featureName: "History",
    elements: {
      count: section.querySelector<HTMLElement>("#history-count")!,
      sentinel: section.querySelector<HTMLElement>("#history-more")!,
    },
    rows: timeline,
    createRow: createEntryElement,
    renderEmptyState: (isEmpty) => {
      if (isEmpty) {
        daysEl.innerHTML = `<p class="history-empty">Nothing here yet. Hold <b>${escapeHtml(shortcut)}</b> anywhere and talk.</p>`;
      }
    },
    isDestroyed: () => destroyed,
  });

  function noMatchesElement(): HTMLElement {
    const el = document.createElement("p");
    el.className = "history-empty";
    el.textContent = "No matches";
    return el;
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
      const resp = await api.searchHistory(q, PAGE_SIZE);
      claim.replaceRows(
        resp.entries.length === 0
          ? [noMatchesElement()]
          : resp.entries.map((entry) => createEntryElement(entry))
      );
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
      if (!value) {
        searchHint.textContent = "";
        void list.load();
      } else {
        void runSearch(value);
      }
    }, SEARCH_DEBOUNCE_MS);
  });

  function createEntryElement(entry: HistoryEntry): HTMLElement {
    const el = document.createElement("article");
    el.className = `entry entry--${entry.source}`;
    el.tabIndex = 0;
    el.dataset.id = entry.id;

    const textHtml = entry.highlighted_text
      ? entry.highlighted_text
      : escapeHtml(entry.text).replace(/\n/g, "<br>");

    el.innerHTML = `
      <span class="entry-dot">${icon(SOURCE_ICONS[entry.source], "small")}</span>
      <p class="entry-text">${textHtml}</p>
      <button type="button" class="entry-more" data-action="expand">${SHOW_MORE}</button>
      <div class="entry-meta">${metaLine(entry)}${sourceBadge(entry)}<span class="entry-actions">
        <button type="button" data-action="copy" aria-label="Copy">${icon("copy", "small")}</button>
        <button type="button" data-action="delete" aria-label="Delete">${icon("x", "small")}</button>
      </span></div>
    `;

    el.addEventListener("click", async (e) => {
      const target = e.target as HTMLElement;
      const button = target.closest<HTMLButtonElement>("button[data-action]");
      const onLongText = target.closest(".entry-text") && el.classList.contains("entry--long");
      if (onLongText || button?.dataset.action === "expand") {
        const expanded = el.classList.toggle("entry--expanded");
        el.querySelector(".entry-more")!.textContent = expanded ? SHOW_LESS : SHOW_MORE;
        return;
      }
      if (!button) return;

      if (button.dataset.action === "copy") {
        const copied = await copyToClipboard(entry.text);
        button.innerHTML = icon(copied ? "check" : "alert", "small");
        button.setAttribute("aria-label", copied ? "Copied" : "Copy failed");
        window.setTimeout(() => {
          button.innerHTML = icon("copy", "small");
          button.setAttribute("aria-label", "Copy");
        }, COPIED_FLASH_MS);
      } else {
        try {
          await api.deleteHistoryEntry(entry.id);
          timeline.rowRemoved(el);
          list.entryRemoved();
        } catch (err) {
          console.error(err);
        }
      }
    });

    return el;
  }

  function startPolling(): void {
    if (pollTimer === null) pollTimer = window.setInterval(() => void list.loadNewer(), NEWER_POLL_MS);
  }

  function stopPolling(): void {
    if (pollTimer !== null) window.clearInterval(pollTimer);
    pollTimer = null;
  }

  void list.load();
  startPolling();

  return {
    destroy() {
      destroyed = true;
      stopPolling();
      list.disconnect();
      textFit.disconnect();
      if (debounceTimer !== null) window.clearTimeout(debounceTimer);
    },
    releaseResources: stopPolling,
    resumeResources() {
      void list.loadNewer();
      startPolling();
    },
  };
}
