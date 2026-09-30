/**
 * The words-per-day chart: a column per day, weekends dimmer, the best day ringed
 * with a callout, a dashed average and a tooltip per column. The card with its
 * 7d / 30d switch is built once; `draw` replaces only the days, the delta and the axis.
 * The plot is a size container, so the callout and tooltip stay inside it in CSS.
 */
import type { ChartSpan, Insights } from "../../api";
import { renderSegmented } from "../../ui/controls";
import { icon } from "../../ui/icons";
import { escapeHtml } from "../html";

export type ChartFigures = Pick<Insights, "days" | "previous_period_words">;

export interface WordsChart {
  draw(figures: ChartFigures, span: ChartSpan): void;
}

const HEADROOM = 1.12;
const LOWEST_BAR_PERCENT = 3;
const GROW_START_MS = 20;
const GROW_STAGGER_MS = 14;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const SPANS = [
  { value: "7", label: "7d" },
  { value: "30", label: "30d" },
] as const;

function localDay(isoDate: string): Date {
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(year, month - 1, day);
}

export function isWeekend(isoDate: string): boolean {
  const weekday = localDay(isoDate).getDay();
  return weekday === 0 || weekday === 6;
}

export function dayLabel(isoDate: string): string {
  const day = localDay(isoDate);
  return `${day.getDate()} ${MONTHS[day.getMonth()]}`;
}

/** The change against the previous period in whole percent, or `null` when that period is empty. */
export function deltaLabel(words: number, previous: number, span: ChartSpan): string | null {
  if (previous === 0) return null;
  const percent = Math.round(((words - previous) / previous) * 100);
  const sign = percent > 0 ? "+" : percent < 0 ? "−" : "";
  return `${sign}${Math.abs(percent)}% vs previous ${span} days`;
}

export function mountWordsChart(
  host: HTMLElement,
  span: ChartSpan,
  onSpan: (span: ChartSpan) => void,
): WordsChart {
  host.innerHTML = `
    <div class="group-label">${icon("trend")}HOW MUCH YOU TALK</div>
    <section class="card chart" aria-label="Words per day">
      <div class="chart-top"><b>Words per day</b><span class="chart-delta"></span><div class="chart-range"></div></div>
      <div class="chart-plot"></div>
      <div class="chart-axis"></div>
    </section>
  `;
  renderSegmented(host.querySelector<HTMLElement>(".chart-range")!, SPANS, `${span}`, (value) =>
    onSpan(Number(value) as ChartSpan),
  );
  const delta = host.querySelector<HTMLElement>(".chart-delta")!;
  const plot = host.querySelector<HTMLElement>(".chart-plot")!;
  const axis = host.querySelector<HTMLElement>(".chart-axis")!;
  return {
    draw(figures, drawnSpan) {
      const words = figures.days.map((day) => day.words);
      const change = deltaLabel(sum(words), figures.previous_period_words, drawnSpan);
      delta.hidden = change === null;
      delta.innerHTML = change === null ? "" : `${change.startsWith("+") ? icon("trend", "small") : ""}${change}`;
      drawPlot(plot, figures.days, words);
      const labels = figures.days.map((day) => dayLabel(day.date));
      axis.innerHTML = [labels[0], labels[Math.floor(labels.length / 2)], labels[labels.length - 1]]
        .map((label) => `<span class="num">${label}</span>`)
        .join("");
    },
  };
}

function drawPlot(plot: HTMLElement, days: ChartFigures["days"], words: number[]): void {
  const most = Math.max(...words);
  const top = Math.max(most, 1) * HEADROOM;
  const percentOf = (value: number) => (value / top) * 100;
  const centerOf = (index: number) => (((index + 0.5) / days.length) * 100).toFixed(2);
  const best = words.filter((value) => value > 0).length >= 2 ? words.indexOf(most) : -1;
  const average = Math.round(sum(words) / words.length);
  plot.innerHTML = `
    ${average > 0 ? `<div class="chart-average" style="bottom:${percentOf(average).toFixed(2)}%"><b class="num">avg ${average}</b></div>` : ""}
    <div class="chart-tip" role="status"></div>
  `;
  const tip = plot.querySelector<HTMLElement>(".chart-tip")!;
  days.forEach((day, index) => {
    const bar = plot.ownerDocument.createElement("div");
    bar.className = `chart-bar${isWeekend(day.date) ? " chart-bar--weekend" : ""}${index === best ? " chart-bar--best" : ""}`;
    bar.tabIndex = 0;
    bar.setAttribute("aria-label", `${number(day.words)} words, ${dayLabel(day.date)}`);
    bar.innerHTML = `<i style="height:0"></i>`;
    const show = () => {
      tip.innerHTML = `<b class="num">${number(day.words)} words</b><span class="num">${dayLabel(day.date)}</span>`;
      tip.style.setProperty("--center", centerOf(index));
      tip.classList.add("chart-tip--visible");
    };
    const hide = () => tip.classList.remove("chart-tip--visible");
    bar.addEventListener("mouseenter", show);
    bar.addEventListener("focus", show);
    bar.addEventListener("mouseleave", hide);
    bar.addEventListener("blur", hide);
    plot.append(bar);
    setTimeout(() => {
      bar.querySelector("i")!.style.height = `${Math.max(LOWEST_BAR_PERCENT, percentOf(day.words)).toFixed(2)}%`;
    }, GROW_START_MS + index * GROW_STAGGER_MS);
  });
  if (best < 0) return;
  const callout = plot.ownerDocument.createElement("div");
  callout.className = "chart-callout";
  callout.style.setProperty("--center", centerOf(best));
  callout.style.bottom = `${percentOf(most).toFixed(2)}%`;
  callout.innerHTML = `<span class="num">${number(most)} words · ${escapeHtml(dayLabel(days[best].date))}</span>`;
  plot.append(callout);
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function number(value: number): string {
  return value.toLocaleString("en-US");
}
