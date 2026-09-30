/**
 * YOUR FAVOURITE WORDS: the three words you say most on a podium with their counts,
 * ranks four to ten as bars measured against the first. The bars grow from zero once
 * mounted. Nothing is mounted before a first word.
 */
import type { WordCount } from "../../api";
import { icon } from "../../ui/icons";
import { escapeHtml } from "../html";

export const FAVOURITE_WORDS = 10;

const PODIUM = 3;
const GROW_START_MS = 20;

export function mountFavouriteWords(host: HTMLElement, items: WordCount[]): void {
  if (items.length === 0) return;
  const most = items[0].count;
  const section = host.ownerDocument.createElement("div");
  section.innerHTML = `
    <div class="group-label">${icon("chart")}YOUR FAVOURITE WORDS</div>
    <section class="card favourite-words" aria-label="Your favourite words">
      <div class="podium">${items.slice(0, PODIUM).map(podiumItem).join("")}</div>
      ${items.length > PODIUM ? `<div class="word-list">${items.slice(PODIUM).map(wordRow).join("")}</div>` : ""}
    </section>
  `;
  host.append(section);
  const fills = [...section.querySelectorAll<HTMLElement>(".word-row-track i")];
  setTimeout(() => {
    fills.forEach((fill, index) => {
      fill.style.width = `${((items[PODIUM + index].count / most) * 100).toFixed(1)}%`;
    });
  }, GROW_START_MS);
}

function podiumItem({ word, count }: WordCount, index: number): string {
  return `
    <div class="podium-item">
      <span class="podium-rank num">${index + 1}</span>
      <div class="podium-word">${escapeHtml(word)}</div>
      <div class="podium-count"><b class="num">${number(count)}</b> ${count === 1 ? "time" : "times"}</div>
    </div>
  `;
}

function wordRow({ word, count }: WordCount, index: number): string {
  return `
    <div class="word-row">
      <span class="word-row-rank num">${index + PODIUM + 1}</span>
      <span class="word-row-word">${escapeHtml(word)}</span>
      <span class="word-row-track"><i style="width:0"></i></span>
      <span class="word-row-count num">${number(count)}</span>
    </div>
  `;
}

function number(value: number): string {
  return value.toLocaleString("en-US");
}
