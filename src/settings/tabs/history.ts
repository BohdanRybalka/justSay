import {
  api,
  MalformedResponseError,
  SidecarTooOldError,
  type HistoryEntry,
  type UserSettings,
} from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import { copyToClipboard } from "../../clipboard";
import { renderSegmented } from "../../ui/controls";
import { icon, type IconName } from "../../ui/icons";
import {
  createHistoryList,
  sidecarTooOldText,
  SENTINEL_READING,
  type BuiltRow,
  type HistoryRowsClaim,
} from "../history-list";
import { countOf, createTimelineRows, formatClock, formatDuration, matchDayGroups } from "../history-timeline";
import { escapeHtml } from "../html";
import type { TabLifecycle } from "../settings";

const SEARCH_DEBOUNCE_MS = 300;
export const NEWER_POLL_MS = 5000;
const PAGE_SIZE = 30;
const COPIED_FLASH_MS = 1500;
const EXCERPT_LEAD_CHARS = 80;

const SOURCE_ICONS: Record<HistoryEntry["source"], IconName> = {
  dictation: "mic",
  file: "file",
  meeting: "users",
};

type HistoryFilter = "all" | "starred";

const FILTER_OPTIONS = [
  { value: "all", label: "All" },
  { value: "starred", label: "Starred" },
] as const;

const SKELETON_CARD = `<div class="entry entry--skeleton" aria-hidden="true"><span class="entry-dot"></span><i></i><i></i></div>`;

/** The panel's title and subtitle, which sit above whatever the panel hosts before the timeline. */
export function renderHistoryHeading(container: HTMLElement): void {
  container.insertAdjacentHTML(
    "beforeend",
    `<h2 class="panel-title">History</h2>
    <p class="panel-subtitle">Everything you've said, kept on this machine.</p>`,
  );
}

/** Marks a collapsed card whose text is cut off, which is what shows its "Show more".
 *  Each text is watched on its own, so a new card, a collapse and a resize are all measured;
 *  a text that has left the page is let go. */
function markLongTexts(records: readonly ResizeObserverEntry[], observer: ResizeObserver): void {
  for (const { target } of records) {
    if (!target.isConnected) {
      observer.unobserve(target);
      continue;
    }
    const card = target.closest<HTMLElement>(".entry");
    if (!card || card.classList.contains("entry--expanded")) continue;
    const cutOff = target.scrollHeight > target.clientHeight + 1;
    card.classList.toggle("entry--long", cutOff || card.classList.contains("entry--excerpt"));
  }
}

/** The backend's marked text starting a few words before its first mark, or `null` when the
 *  first mark already sits near the start. Cut on the parsed nodes, so no entity is split. */
export function matchExcerpt(highlighted: string): string | null {
  const template = document.createElement("template");
  template.innerHTML = highlighted;
  const lead = template.content.firstChild;
  if (lead?.nodeType !== Node.TEXT_NODE || lead.nextSibling?.nodeName !== "MARK") return null;
  const text = lead.textContent ?? "";
  if (text.length <= EXCERPT_LEAD_CHARS) return null;
  const tail = text.slice(-EXCERPT_LEAD_CHARS);
  lead.textContent = `…${tail.slice(tail.search(/\s/) + 1)}`;
  return template.innerHTML;
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
    <div class="search-box">${icon("search")}<input type="search" id="history-search" placeholder="Search what you said" aria-label="Search" /><div id="history-filter" aria-label="Show"></div></div>
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
  const shortcut = formatAccelerator(settings.shortcut, detectShortcutPlatform(navigator));

  let searchClaim: HistoryRowsClaim | null = null;
  let debounceTimer: number | null = null;
  let pollTimer: number | null = null;
  let destroyed = false;
  let starredOnly = false;
  let openMenu: { menu: HTMLElement; trigger: HTMLButtonElement } | null = null;

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
      if (!isEmpty) return;
      daysEl.innerHTML = starredOnly
        ? `<p class="history-empty">Nothing starred yet.</p>`
        : `<p class="history-empty">Nothing here yet. Hold <b>${escapeHtml(shortcut)}</b> anywhere and talk.</p>`;
    },
    isDestroyed: () => destroyed,
    starredOnly: () => starredOnly,
  });

  function showCurrentView(): void {
    const query = searchInput.value.trim();
    if (query) void runSearch(query);
    else void list.load();
  }

  renderSegmented<HistoryFilter>(
    section.querySelector<HTMLElement>("#history-filter")!,
    FILTER_OPTIONS,
    "all",
    (filter) => {
      starredOnly = filter === "starred";
      showCurrentView();
    },
  );

  function noMatchesElement(): HTMLElement {
    const el = document.createElement("p");
    el.className = "history-empty";
    el.textContent = "Nothing you said matches that.";
    return el;
  }

  /** Exact matches by day, then everything looser under "Close matches". */
  function searchTiers(rows: readonly BuiltRow[]): HTMLElement[] {
    const painted = matchDayGroups(rows.filter((row) => row.entry.match === "exact"));
    const close = rows.filter((row) => row.entry.match !== "exact");
    if (close.length > 0) {
      const heading = document.createElement("h3");
      heading.className = "search-tier";
      heading.textContent = "Close matches";
      painted.push(heading, ...matchDayGroups(close));
    }
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
      const resp = await api.searchHistory(q, PAGE_SIZE, starredOnly);
      const built = resp.entries.map((entry) => ({ entry, element: createEntryElement(entry) }));
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

  function closeMenu(returnFocus = false): void {
    if (!openMenu) return;
    const { menu, trigger } = openMenu;
    openMenu = null;
    menu.remove();
    trigger.setAttribute("aria-expanded", "false");
    trigger.closest(".entry")?.classList.remove("entry--menu-open");
    document.removeEventListener("pointerdown", closeMenuOutside, true);
    document.removeEventListener("keydown", closeMenuOnEscape, true);
    if (returnFocus) trigger.focus();
  }

  function closeMenuOutside(event: Event): void {
    const target = event.target as Node;
    if (openMenu && !openMenu.menu.contains(target) && !openMenu.trigger.contains(target)) {
      closeMenu();
    }
  }

  function closeMenuOnEscape(event: KeyboardEvent): void {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeMenu(true);
  }

  /** The card's "more" menu, under its trigger; one menu is open at a time. */
  function toggleMenu(trigger: HTMLButtonElement): void {
    const wasOpen = openMenu?.trigger === trigger;
    closeMenu();
    if (wasOpen) return;
    const menu = document.createElement("div");
    menu.className = "entry-menu";
    menu.setAttribute("role", "menu");
    menu.innerHTML = `<button type="button" role="menuitem" data-action="delete">${icon("x", "small")}Delete</button>`;
    trigger.after(menu);
    trigger.setAttribute("aria-expanded", "true");
    trigger.closest(".entry")?.classList.add("entry--menu-open");
    openMenu = { menu, trigger };
    document.addEventListener("pointerdown", closeMenuOutside, true);
    document.addEventListener("keydown", closeMenuOnEscape, true);
    menu.querySelector("button")!.focus();
  }

  function paintStar(button: HTMLButtonElement, starred: boolean): void {
    button.setAttribute("aria-pressed", String(starred));
    button.classList.toggle("entry-star--on", starred);
  }

  function createEntryElement(entry: HistoryEntry): HTMLElement {
    const el = document.createElement("article");
    el.className = `entry entry--${entry.source}`;
    el.tabIndex = 0;
    el.dataset.id = entry.id;

    const textHtml = entry.highlighted_text
      ? entry.highlighted_text
      : escapeHtml(entry.text).replace(/\n/g, "<br>");
    const excerpt = entry.highlighted_text ? matchExcerpt(entry.highlighted_text) : null;
    if (excerpt !== null) el.classList.add("entry--excerpt");
    const shownHtml =
      excerpt === null
        ? textHtml
        : `<span class="entry-full">${textHtml}</span><span class="entry-excerpt">${excerpt}</span>`;

    el.innerHTML = `
      <span class="entry-dot">${icon(SOURCE_ICONS[entry.source], "small")}</span>
      <div class="entry-body">
        <p class="entry-text">${shownHtml}<button type="button" class="entry-less" data-action="expand" aria-label="Show less">less</button></p>
        <button type="button" class="entry-more" data-action="expand" aria-label="Show more">… more</button>
      </div>
      <div class="entry-meta">${metaLine(entry)}${sourceBadge(entry)}<span class="entry-actions">
        <button type="button" data-action="copy" aria-label="Copy">${icon("copy", "small")}</button>
        <button type="button" class="entry-star" data-action="star" aria-label="Star">${icon("star", "small")}</button>
        <button type="button" data-action="more" aria-label="More" aria-haspopup="menu" aria-expanded="false">${icon("dots", "small")}</button>
      </span></div>
    `;

    textFit.observe(el.querySelector(".entry-text")!);
    let starred = entry.starred;
    let starSaving = false;
    paintStar(el.querySelector<HTMLButtonElement>('[data-action="star"]')!, starred);

    el.addEventListener("click", async (e) => {
      const target = e.target as HTMLElement;
      const button = target.closest<HTMLButtonElement>("button[data-action]");
      const onLongText = target.closest(".entry-text") && el.classList.contains("entry--long");
      if (onLongText || button?.dataset.action === "expand") {
        el.classList.toggle("entry--expanded");
        return;
      }
      if (!button) return;

      if (button.dataset.action === "more") {
        toggleMenu(button);
      } else if (button.dataset.action === "star") {
        if (starSaving) return;
        starSaving = true;
        paintStar(button, !starred);
        try {
          await api.setHistoryStarred(entry.id, !starred);
          starred = !starred;
        } catch (err) {
          paintStar(button, starred);
          console.error(err);
        } finally {
          starSaving = false;
        }
      } else if (button.dataset.action === "copy") {
        const copied = await copyToClipboard(entry.text);
        button.innerHTML = icon(copied ? "check" : "alert", "small");
        button.setAttribute("aria-label", copied ? "Copied" : "Copy failed");
        window.setTimeout(() => {
          button.innerHTML = icon("copy", "small");
          button.setAttribute("aria-label", "Copy");
        }, COPIED_FLASH_MS);
      } else if (button.dataset.action === "delete") {
        closeMenu();
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
      closeMenu();
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
