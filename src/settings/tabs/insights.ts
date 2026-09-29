/**
 * The Insights panel: a greeting with today's dictation, then the clay card of the
 * time talking saved this month. The backend computes every figure; this panel
 * formats them, and reads them again each time the window comes back on screen.
 */
import { api, type Insights } from "../../api";
import { detectShortcutPlatform, formatAccelerator } from "../../accelerator";
import { formatCoarseDuration, formatHoursClock, wholeMinutes } from "../../format";
import { escapeHtml } from "../html";
import type { TabLifecycle } from "../settings";

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

  async function read(): Promise<void> {
    const token = ++latestRead;
    const greeting = greetingFor(new Date().getHours());
    title.textContent = firstName ? `${greeting}, ${firstName}` : greeting;
    try {
      const figures = await api.insights();
      if (token !== latestRead) return;
      today.innerHTML = todayLine(figures.today, shortcut);
      body.innerHTML = figures.month.recordings > 0 ? savedCard(figures) : "";
    } catch (e) {
      if (token !== latestRead) return;
      console.error("Reading insights failed:", e);
      body.innerHTML = `<p class="panel-error">Insights could not be read. <button type="button" class="btn btn-small">Try again</button></p>`;
      body.querySelector("button")!.addEventListener("click", () => void read());
    }
  }

  const disown = () => {
    latestRead += 1;
  };

  if (!windowHidden) void read();
  return { destroy: disown, releaseResources: disown, resumeResources: () => void read() };
}

function todayLine(today: Insights["today"], shortcut: string): string {
  const keys = `<b class="num">${escapeHtml(shortcut)}</b>`;
  if (today.recordings === 0) return `Hold ${keys} anywhere and talk.`;
  return `${count(today.words, "word")} today across ${count(today.recordings, "recording")}. Hold ${keys} anywhere to add more.`;
}

function savedCard({ month, streak }: Insights): string {
  const typing = wholeMinutes(month.typing_seconds);
  const speaking = wholeMinutes(month.speaking_seconds);
  const saved = Math.max(0, typing - speaking);
  const longest = streak.current_days > 1 && streak.current_days === streak.longest_days;
  return `
    <section class="saved-card" aria-label="Time saved this month">
      <div class="saved-card-label">YOU SAVED THIS MONTH</div>
      <div class="saved-card-value num">${formatCoarseDuration(saved * 60)}</div>
      <p class="saved-card-note">That's what typing these words by hand would have cost you. ${savedPhrase(saved)}</p>
      ${month.speaking_seconds >= COMPARE_FROM_SECONDS ? compareBars(typing, speaking) : ""}
      <div class="saved-figures">
        <div><b class="num">${number(month.words)}</b><span>words in ${count(month.recordings, "recording", false)}</span></div>
        <div><b class="num">${count(streak.current_days, "day", false)}</b><span>streak${longest ? " · your longest" : ""}</span></div>
        ${month.pace_wpm === null ? "" : `<div><b class="num">${month.pace_wpm} wpm</b><span>${paceLine(month.typing_seconds / month.speaking_seconds)}</span></div>`}
      </div>
    </section>
  `;
}

function compareBars(typing: number, speaking: number): string {
  const longer = Math.max(typing, speaking, 1);
  const row = (label: string, minutes: number, dim: boolean) => `
    <div class="saved-compare-row"><span>${label}</span><span class="saved-compare-track${dim ? " saved-compare-track--dim" : ""}"><i style="width:${((minutes / longer) * 100).toFixed(1)}%"></i></span><b class="num">${formatHoursClock(minutes * 60)}</b></div>`;
  return `<div class="saved-compare">${row("Typing", typing, true)}${row("Speaking", speaking, false)}</div>`;
}

function number(value: number): string {
  return value.toLocaleString("en-US");
}

function count(value: number, noun: string, bold = true): string {
  const figure = number(value);
  return `${bold ? `<b class="num">${figure}</b>` : figure} ${noun}${value === 1 ? "" : "s"}`;
}
