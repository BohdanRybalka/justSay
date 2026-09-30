/**
 * YOUR FAVOURITE WORDS: the three words you say most on a podium with their counts,
 * ranks four to ten as bars measured against the first, fillers flagged in clay. The
 * All / Fillers switch appears once a filler is said; `draw` replaces the podium, the
 * list and the note under them. Nothing is mounted before a first word.
 */
import type { FillerNote, TopWordsResponse, WordCount, WordFilter } from "../../api";
import { renderSegmented } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { escapeHtml } from "../html";

export const FAVOURITE_WORDS = 10;

export interface FavouriteWords {
  draw(top: TopWordsResponse): void;
}

const PODIUM = 3;
const GROW_START_MS = 20;
const FILTERS = [
  { value: "all", label: "All" },
  { value: "fillers", label: "Fillers" },
] as const;
const NUMBER_WORDS = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
];

/** Up to twenty in words, as the design writes them; larger numbers as figures. */
export function numberWord(value: number): string {
  return NUMBER_WORDS[value] ?? number(value);
}

/** How often a filler comes, to the nearest half minute, or hour past ninety minutes. */
export function frequencyPhrase(minutes: number): string {
  if (minutes < 0.75) return "more than once a minute";
  const [span, unit] = minutes < 90 ? [minutes, "minute"] : [minutes / 60, "hour"];
  const halves = Math.round(span * 2);
  if (halves === 2) return "about once a minute";
  const whole = numberWord(Math.floor(halves / 2));
  return `about once every ${halves % 2 === 0 ? whole : `${whole} and a half`} ${unit}s`;
}

export function fillerNote(note: FillerNote): string {
  const said = `You said <b>${escapeHtml(note.word)}</b> ${times(note.count)}`;
  const first = note.minutes_between === null ? `${said}.` : `${said} — ${frequencyPhrase(note.minutes_between)}.`;
  const { fillers_in_top: fillers, top_size: top } = note;
  if (fillers === 0 || top < 2) return first;
  const share = fillers === top ? "All" : capitalised(numberWord(fillers));
  const verb = fillers === 1 ? "is a filler word" : "are filler words";
  return `${first} ${share} of your top ${numberWord(top)} ${verb}.`;
}

export function mountFavouriteWords(
  host: HTMLElement,
  top: TopWordsResponse,
  filter: WordFilter,
  onFilter: (filter: WordFilter) => void,
): FavouriteWords | null {
  if (top.items.length === 0) return null;
  host.innerHTML = `
    <div class="group-label">${icon("chart")}YOUR FAVOURITE WORDS<div class="words-filter"></div></div>
    <section class="card favourite-words" aria-label="Your favourite words"></section>
  `;
  if (top.note !== null) {
    renderSegmented(host.querySelector<HTMLElement>(".words-filter")!, FILTERS, filter, onFilter);
  }
  const card = host.querySelector<HTMLElement>(".favourite-words")!;
  const draw = ({ items, note }: TopWordsResponse): void => {
    card.innerHTML = `
      <div class="podium">${items.slice(0, PODIUM).map(podiumItem).join("")}</div>
      ${items.length > PODIUM ? `<div class="word-list">${items.slice(PODIUM).map(wordRow).join("")}</div>` : ""}
      ${note === null ? "" : `<div class="words-note">${icon("sparkle")}<div>${fillerNote(note)}</div></div>`}
    `;
    const fills = [...card.querySelectorAll<HTMLElement>(".word-row-track i")];
    setTimeout(() => {
      fills.forEach((fill, index) => {
        fill.style.width = `${((items[PODIUM + index].count / items[0].count) * 100).toFixed(1)}%`;
      });
    }, GROW_START_MS);
  };
  draw(top);
  return { draw };
}

function podiumItem({ word, count, is_filler }: WordCount, index: number): string {
  return `
    <div class="podium-item${is_filler ? " podium-item--filler" : ""}">
      <span class="podium-rank num">${index + 1}</span>
      <div class="podium-word">${escapeHtml(word)}</div>
      <div class="podium-count"><b class="num">${number(count)}</b> ${count === 1 ? "time" : "times"}${is_filler ? " · filler" : ""}</div>
    </div>
  `;
}

function wordRow({ word, count, is_filler }: WordCount, index: number): string {
  return `
    <div class="word-row">
      <span class="word-row-rank num">${index + PODIUM + 1}</span>
      <span class="word-row-word">${escapeHtml(word)}${is_filler ? ` <span class="chip">filler</span>` : ""}</span>
      <span class="word-row-track"><i${is_filler ? ` class="word-row-fill--filler"` : ""} style="width:0"></i></span>
      <span class="word-row-count num">${number(count)}</span>
    </div>
  `;
}

function times(count: number): string {
  if (count === 1) return "once";
  if (count === 2) return "twice";
  return `${number(count)} times`;
}

function capitalised(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function number(value: number): string {
  return value.toLocaleString("en-US");
}
