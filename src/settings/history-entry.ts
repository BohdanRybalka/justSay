/**
 * One recording as a card, the same in History and in the Meetings and Files sections:
 * its text, its meta line, and Copy, Star and the menu with Delete.
 */
import { api, type HistoryEntry } from "../api";
import { copyToClipboard } from "../clipboard";
import { icon, type IconName } from "../ui/icons";
import { countOf, formatClock, formatDuration } from "./history-timeline";
import { escapeHtml } from "./html";

const COPIED_FLASH_MS = 1500;
const EXCERPT_LEAD_CHARS = 80;

export const SOURCE_ICONS: Record<HistoryEntry["source"], IconName> = {
  dictation: "mic",
  file: "file",
  meeting: "users",
};

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

function paintStar(button: HTMLButtonElement, starred: boolean): void {
  button.setAttribute("aria-pressed", String(starred));
  button.classList.toggle("entry-star--on", starred);
}

export interface EntryCards {
  create(entry: HistoryEntry): HTMLElement;
  destroy(): void;
}

/** Builds the cards of one list. One menu is open at a time; `deleted` runs with a card
 *  once the backend has deleted its entry. */
export function createEntryCards(deleted: (card: HTMLElement) => void): EntryCards {
  const textFit = new ResizeObserver(markLongTexts);
  let openMenu: { menu: HTMLElement; trigger: HTMLButtonElement } | null = null;

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

  function create(entry: HistoryEntry): HTMLElement {
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
          deleted(el);
        } catch (err) {
          console.error(err);
        }
      }
    });

    return el;
  }

  return {
    create,
    destroy() {
      closeMenu();
      textFit.disconnect();
    },
  };
}
