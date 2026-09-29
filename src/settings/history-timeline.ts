import type { HistoryDay, HistoryEntry } from "../api";
import type { BuiltRow, HistoryRows } from "./history-list";
import { escapeHtml } from "./html";

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];
const UNKNOWN_DAY = "unknown";

/** Numbers as the design writes them, whatever the OS locale: `7,142`. */
export function formatNumber(value: number): string {
  return value.toLocaleString("en-US");
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** `YYYY-MM-DD` on this machine's clock, the key the backend's `days` use. */
export function localDayKey(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function dayKeyOf(entry: HistoryEntry): string | null {
  return entry.timestamp == null ? null : localDayKey(new Date(entry.timestamp));
}

/** "Today", "Yesterday", "Monday, 28 July", with the year only when it is not this one. */
export function formatDayLabel(key: string | null, now: Date): string {
  if (key === null) return "Date unknown";
  if (key === localDayKey(now)) return "Today";
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
  if (key === localDayKey(yesterday)) return "Yesterday";
  const [year, month, day] = key.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  const label = `${WEEKDAYS[date.getDay()]}, ${day} ${MONTHS[month - 1]}`;
  return year === now.getFullYear() ? label : `${label} ${year}`;
}

export function formatClock(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** `m:ss`, and `h:mm:ss` from an hour on. */
export function formatDuration(seconds: number): string {
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = pad(total % 60);
  return hours > 0 ? `${hours}:${pad(minutes)}:${rest}` : `${minutes}:${rest}`;
}

export function countOf(value: number, singular: string, plural: string): string {
  return `<span class="num">${formatNumber(value)}</span> ${value === 1 ? singular : plural}`;
}

function dayHeadHtml(key: string | null, today: Date, counts: readonly string[]): string {
  return (
    `<span class="day-head-dot"></span><b>${escapeHtml(formatDayLabel(key, today))}</b>` +
    counts.map((count) => `<span>·</span><span>${count}</span>`).join("")
  );
}

function paintMatchHead(root: HTMLElement, today: Date): void {
  const key = root.dataset.day || null;
  const matches = root.querySelector(".day-body")!.children.length;
  root.querySelector(".day-head")!.innerHTML = dayHeadHtml(key, today, [
    countOf(matches, "match", "matches"),
  ]);
}

function newestFirst(a: BuiltRow, b: BuiltRow): number {
  const at = (row: BuiltRow) =>
    row.entry.timestamp == null ? -Infinity : new Date(row.entry.timestamp).getTime();
  return at(b) - at(a);
}

/** Search results as day groups, newest day first, each header counting the matches under it. */
export function matchDayGroups(rows: readonly BuiltRow[], now: Date = new Date()): HTMLElement[] {
  const groups = new Map<string, HTMLElement>();
  for (const row of [...rows].sort(newestFirst)) {
    const key = dayKeyOf(row.entry);
    let root = groups.get(key ?? UNKNOWN_DAY);
    if (!root) {
      root = document.createElement("section");
      root.className = "day-group";
      root.dataset.day = key ?? "";
      root.innerHTML = `<div class="day-head"></div><div class="day-body"></div>`;
      groups.set(key ?? UNKNOWN_DAY, root);
    }
    root.querySelector(".day-body")!.append(row.element);
  }
  for (const root of groups.values()) paintMatchHead(root, now);
  return [...groups.values()];
}

export interface TimelineRows extends HistoryRows {
  /** Takes one card off the timeline, and its day with it once the day is empty. */
  rowRemoved(element: HTMLElement): void;
}

interface DayGroup {
  key: string | null;
  root: HTMLElement;
  head: HTMLElement;
  body: HTMLElement;
}

/**
 * The day groups inside `container`. A header reads its day's totals from the
 * backend's `days`, which count the whole day; only a day those never named
 * falls back to the cards on screen.
 */
export function createTimelineRows(container: HTMLElement, now: () => Date = () => new Date()): TimelineRows {
  const groups = new Map<string, DayGroup>();
  const totals = new Map<string, HistoryDay>();

  function reset(): void {
    groups.clear();
    totals.clear();
    container.replaceChildren();
  }

  function groupFor(key: string | null, atTop: boolean): DayGroup {
    const existing = groups.get(key ?? UNKNOWN_DAY);
    if (existing) return existing;
    const root = document.createElement("section");
    root.className = "day-group";
    const head = document.createElement("div");
    head.className = "day-head";
    const body = document.createElement("div");
    body.className = "day-body";
    root.append(head, body);
    if (atTop) container.prepend(root);
    else container.append(root);
    const created = { key, root, head, body };
    groups.set(key ?? UNKNOWN_DAY, created);
    return created;
  }

  function adoptDays(days: readonly HistoryDay[]): void {
    for (const day of days) totals.set(day.date ?? UNKNOWN_DAY, day);
  }

  function paintHeads(): void {
    const today = now();
    for (const [mapKey, group] of groups) {
      const cards = Array.from(group.body.children) as HTMLElement[];
      const day = totals.get(mapKey) ?? {
        date: group.key,
        recordings: cards.length,
        words: cards.reduce((sum, card) => sum + Number(card.dataset.words ?? 0), 0),
      };
      group.head.innerHTML = dayHeadHtml(group.key, today, [
        countOf(day.recordings, "recording", "recordings"),
        countOf(day.words, "word", "words"),
      ]);
    }
  }

  function place(row: BuiltRow, atTop: boolean): void {
    row.element.dataset.words = String(row.entry.word_count ?? 0);
    const body = groupFor(dayKeyOf(row.entry), atTop).body;
    if (atTop) body.prepend(row.element);
    else body.append(row.element);
  }

  function append(rows: readonly BuiltRow[], days: readonly HistoryDay[]): void {
    adoptDays(days);
    for (const row of rows) place(row, false);
    paintHeads();
  }

  return {
    replace(rows, days) {
      reset();
      append(rows, days);
    },
    append,
    prepend(rows, days) {
      adoptDays(days);
      for (const row of [...rows].reverse()) place(row, true);
      paintHeads();
    },
    replaceWith(elements) {
      reset();
      container.append(...elements);
    },
    rowRemoved(element) {
      const root = element.closest<HTMLElement>(".day-group");
      element.remove();
      const entry = Array.from(groups).find(([, group]) => group.root === root);
      if (!entry) {
        if (root?.querySelector(".day-body")!.children.length === 0) root.remove();
        else if (root) paintMatchHead(root, now());
        return;
      }
      const [mapKey, group] = entry;
      const day = totals.get(mapKey);
      if (day) {
        totals.set(mapKey, {
          ...day,
          recordings: day.recordings - 1,
          words: day.words - Number(element.dataset.words ?? 0),
        });
      }
      if (group.body.children.length === 0) {
        group.root.remove();
        groups.delete(mapKey);
        totals.delete(mapKey);
      }
      paintHeads();
    },
  };
}
