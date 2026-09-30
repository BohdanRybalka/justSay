/**
 * The Insights panel: a greeting with today's dictation, the clay card of the time
 * talking saved this month, the words-per-day chart, facts about your voice and your
 * favourite words, then the card shared or saved as an image. The backend computes every
 * figure; this panel formats them, and reads them again each time the window comes back
 * on screen. The chart's 7d / 30d switch and the words' All / Fillers switch each read
 * again and redraw their own card alone.
 */
import { api, type ChartSpan, type Insights, type TopWordsResponse, type WordFilter } from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import { formatCoarseDuration, formatHoursClock, wholeMinutes } from "../../format";
import { icon } from "../../ui/icons";
import { escapeHtml } from "../html";
import type { TabLifecycle } from "../settings";
import { mountWordsChart, type WordsChart } from "./insights-chart";
import { copyImage, saveImage, savedCardPng, type SavedCardView } from "./insights-image";
import { voiceFacts } from "./insights-voice";
import { FAVOURITE_WORDS, mountFavouriteWords, type FavouriteWords } from "./insights-words";

export interface InsightsViewer {
  name: string;
  shortcut: string;
}

const COMPARE_FROM_SECONDS = 600;

const SAVED_PHRASES: readonly [belowMinutes: number, phrase: string][] = [
  [10, "A few minutes back."],
  [30, "About a coffee break."],
  [90, "About a long lunch."],
  [240, "Roughly one free afternoon."],
  [480, "About a whole day off."],
  [Infinity, "More than a working day."],
];

export function greetingFor(hour: number): string {
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 17) return "Good afternoon";
  return "Good evening";
}

export function savedPhrase(minutes: number): string {
  return SAVED_PHRASES.find(([below]) => minutes < below)![1];
}

/** How many times typing speed a pace is, said the way the card says it. */
export function paceLine(timesTyping: number): string {
  const times = Math.round(timesTyping);
  if (times <= 1) return "about your typing speed";
  if (times === 2) return "twice your typing";
  if (times === 3) return "three times your typing";
  return `${times} times your typing`;
}

export function renderInsights(
  container: HTMLElement,
  viewer: InsightsViewer,
  windowHidden: boolean,
): TabLifecycle {
  const section = document.createElement("div");
  section.className = "insights";
  const firstName = viewer.name.trim().split(/\s+/)[0];
  section.innerHTML = `
    <h2 class="panel-title"></h2>
    <p class="panel-subtitle" id="insights-today"></p>
    <div id="insights-body"></div>
  `;
  container.append(section);
  const title = section.querySelector<HTMLElement>(".panel-title")!;
  const today = section.querySelector<HTMLElement>("#insights-today")!;
  const body = section.querySelector<HTMLElement>("#insights-body")!;
  const shortcut = formatAccelerator(viewer.shortcut, detectShortcutPlatform(navigator));

  let latestRead = 0;
  let latestChartRead = 0;
  let latestWordsRead = 0;
  let span: ChartSpan = 30;
  let wordFilter: WordFilter = "all";
  let chart: WordsChart | null = null;
  let favouriteWords: FavouriteWords | null = null;

  async function read(): Promise<void> {
    const token = ++latestRead;
    latestChartRead += 1;
    latestWordsRead += 1;
    const requested = span;
    const greeting = greetingFor(new Date().getHours());
    title.textContent = firstName ? `${greeting}, ${firstName}` : greeting;
    try {
      const [figures, favourites] = await Promise.all([api.insights(requested), readWords()]);
      if (token !== latestRead) return;
      today.innerHTML = todayLine(figures.today, shortcut);
      const chartShown = chart !== null;
      const card = figures.month.recordings > 0 ? savedCardView(figures) : null;
      body.innerHTML = card === null ? "" : savedCard(card);
      chart = null;
      if (chartShown || hasSpoken(figures)) mountChart(figures, requested);
      body.insertAdjacentHTML("beforeend", voiceFacts(figures));
      const wordsHost = document.createElement("div");
      body.append(wordsHost);
      favouriteWords = mountFavouriteWords(wordsHost, favourites.top, wordFilter, (next) => void switchWords(next));
      if (favourites.filter !== wordFilter) void switchWords(wordFilter);
      if (card !== null) mountShare(card);
    } catch (e) {
      if (token !== latestRead) return;
      showFailure(e);
    }
  }

  async function readWords(): Promise<{ top: TopWordsResponse; filter: WordFilter }> {
    const filter = wordFilter;
    const top = await api.wordsTop(FAVOURITE_WORDS, filter);
    if (top.note !== null || filter === "all") return { top, filter };
    wordFilter = "all";
    return { top: await api.wordsTop(FAVOURITE_WORDS, "all"), filter: "all" };
  }

  async function switchWords(next: WordFilter): Promise<void> {
    wordFilter = next;
    const token = ++latestWordsRead;
    try {
      const top = await api.wordsTop(FAVOURITE_WORDS, next);
      if (token !== latestWordsRead) return;
      favouriteWords?.draw(top);
    } catch (e) {
      if (token !== latestWordsRead) return;
      showFailure(e);
    }
  }

  function mountShare(card: SavedCardView): void {
    const row = document.createElement("div");
    row.className = "share-month";
    row.innerHTML = `
      <button type="button" class="btn btn-primary">${icon("share", "small")}Share this month</button>
      <button type="button" class="btn">Download as image</button>
      <span class="share-month-status" role="status"></span>
    `;
    body.append(row);
    const [share, download] = row.querySelectorAll("button");
    const status = row.querySelector<HTMLElement>(".share-month-status")!;
    const act = async (work: (png: Uint8Array) => Promise<string>, failure: string): Promise<void> => {
      share.disabled = download.disabled = true;
      status.textContent = "";
      try {
        status.textContent = await work(await savedCardPng(card));
      } catch (e) {
        console.error(`${failure}:`, e);
        status.textContent = `${failure}. Try again`;
      } finally {
        share.disabled = download.disabled = false;
      }
    };
    share.addEventListener("click", () =>
      void act(async (png) => {
        await copyImage(png);
        return "Image copied — paste it anywhere";
      }, "Couldn't copy the image"),
    );
    download.addEventListener("click", () =>
      void act(async (png) => ((await saveImage(png, imageName(new Date()))) ? "Image saved" : ""), "Couldn't save the image"),
    );
  }

  function mountChart(figures: Insights, requested: ChartSpan): void {
    const host = document.createElement("div");
    body.append(host);
    chart = mountWordsChart(host, span, (next) => void switchSpan(next));
    if (requested === span) chart.draw(figures, span);
    else void switchSpan(span);
  }

  async function switchSpan(next: ChartSpan): Promise<void> {
    span = next;
    const token = ++latestChartRead;
    try {
      const figures = await api.insights(next);
      if (token !== latestChartRead) return;
      chart?.draw(figures, next);
    } catch (e) {
      if (token !== latestChartRead) return;
      showFailure(e);
    }
  }

  function showFailure(e: unknown): void {
    console.error("Reading insights failed:", e);
    chart = null;
    body.innerHTML = `<p class="panel-error">Insights could not be read. <button type="button" class="btn btn-small">Try again</button></p>`;
    body.querySelector("button")!.addEventListener("click", () => void read());
  }

  const disown = () => {
    latestRead += 1;
    latestChartRead += 1;
    latestWordsRead += 1;
  };

  if (!windowHidden) void read();
  return { destroy: disown, releaseResources: disown, resumeResources: () => void read() };
}

function hasSpoken({ month, days, previous_period_words }: Insights): boolean {
  return month.recordings > 0 || previous_period_words > 0 || days.some((day) => day.words > 0);
}

function todayLine(today: Insights["today"], shortcut: string): string {
  const keys = `<b class="num">${escapeHtml(shortcut)}</b>`;
  if (today.recordings === 0) return `Hold ${keys} anywhere and talk.`;
  return `${count(today.words, "word")} today across ${count(today.recordings, "recording")}. Hold ${keys} anywhere to add more.`;
}

export function savedCardView({ month, streak }: Insights): SavedCardView {
  const typing = wholeMinutes(month.typing_seconds);
  const speaking = wholeMinutes(month.speaking_seconds);
  const saved = Math.max(0, typing - speaking);
  const longer = Math.max(typing, speaking, 1);
  const longest = streak.current_days > 1 && streak.current_days === streak.longest_days;
  const bar = (label: string, minutes: number, dim: boolean) => ({
    label,
    fraction: minutes / longer,
    time: formatHoursClock(minutes * 60),
    dim,
  });
  return {
    value: formatCoarseDuration(saved * 60),
    note: `That's what typing these words by hand would have cost you. ${savedPhrase(saved)}`,
    compare: month.speaking_seconds >= COMPARE_FROM_SECONDS ? [bar("Typing", typing, true), bar("Speaking", speaking, false)] : [],
    figures: [
      { value: number(month.words), label: `words in ${count(month.recordings, "recording", false)}` },
      { value: count(streak.current_days, "day", false), label: `streak${longest ? " · your longest" : ""}` },
      ...(month.pace_wpm === null
        ? []
        : [{ value: `${month.pace_wpm} wpm`, label: paceLine(month.typing_seconds / month.speaking_seconds) }]),
    ],
  };
}

/** The file name offered when saving the card, after the month it shows. */
export function imageName(now: Date): string {
  return `JustSay-${now.toLocaleString("en-US", { month: "long" })}-${now.getFullYear()}.png`;
}

function savedCard({ value, note, compare, figures }: SavedCardView): string {
  const row = ({ label, fraction, time, dim }: SavedCardView["compare"][number]) => `
    <div class="saved-compare-row"><span>${label}</span><span class="saved-compare-track${dim ? " saved-compare-track--dim" : ""}"><i style="width:${(fraction * 100).toFixed(1)}%"></i></span><b class="num">${time}</b></div>`;
  return `
    <section class="saved-card" aria-label="Time saved this month">
      <div class="saved-card-label">YOU SAVED THIS MONTH</div>
      <div class="saved-card-value num">${value}</div>
      <p class="saved-card-note">${note}</p>
      ${compare.length === 0 ? "" : `<div class="saved-compare">${compare.map(row).join("")}</div>`}
      <div class="saved-figures">
        ${figures.map((figure) => `<div><b class="num">${figure.value}</b><span>${figure.label}</span></div>`).join("")}
      </div>
    </section>
  `;
}

function number(value: number): string {
  return value.toLocaleString("en-US");
}

function count(value: number, noun: string, bold = true): string {
  const figure = number(value);
  return `${bold ? `<b class="num">${figure}</b>` : figure} ${noun}${value === 1 ? "" : "s"}`;
}
