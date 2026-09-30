/**
 * ABOUT YOUR VOICE: four fact cards — the words you use, the hour you talk most, your
 * longest dictation and this week's meetings. A card with nothing to say is left out,
 * and the section with it when all four are.
 */
import type { Insights } from "../../api";
import { formatCoarseDuration, formatElapsedClock, wholeMinutes } from "../../format";
import { icon, type IconName } from "../../ui/icons";

export type VoiceFigures = Pick<Insights, "vocabulary" | "peak_hour" | "longest" | "meetings_week">;

const SHARE_PHRASES: readonly [from: number, below: number, phrase: string][] = [
  [0.21, 0.29, "About a quarter"],
  [0.29, 0.34, "Almost a third"],
  [0.44, 0.56, "About half"],
  [0.56, Infinity, "Most"],
];

/** An hour of the day on a 12-hour clock, "6 PM", as the design writes it. */
export function hourLabel(hour: number): string {
  return `${clockHour(hour)} ${hour % 24 < 12 ? "AM" : "PM"}`;
}

/** A share as the phrase that is true of it, or as whole percent between the phrases. */
export function sharePhrase(share: number): string {
  const band = SHARE_PHRASES.find(([from, below]) => share >= from && share < below);
  return band ? band[2] : `${Math.round(share * 100)}%`;
}

export function voiceFacts({ vocabulary, peak_hour, longest, meetings_week }: VoiceFigures): string {
  const cards = [
    vocabulary > 0 &&
      fact("book", false, `<b class="num">${count(vocabulary, "different word")}</b>`, "Across everything you've said so far."),
    peak_hour !== null &&
      fact(
        "clock",
        true,
        `<b>You talk most at ${hourLabel(peak_hour.hour)}</b>`,
        `${sharePhrase(peak_hour.share)} of everything you dictate happens between ${clockHour(peak_hour.hour - 1)} and ${clockHour(peak_hour.hour + 1)}.`,
      ),
    longest !== null &&
      fact(
        "flame",
        false,
        `<b class="num">${formatElapsedClock(longest.seconds)}</b>`,
        `Your longest run without stopping.${longest.words > 0 ? ` It became ${count(longest.words, "word")}.` : ""}`,
      ),
    meetings_week.count > 0 &&
      fact("users", true, `<b class="num">${count(meetings_week.count, "meeting")}</b>`, meetingsLine(meetings_week.seconds)),
  ].filter((card): card is string => card !== false);
  if (cards.length === 0) return "";
  return `
    <div class="group-label">${icon("sparkle")}ABOUT YOUR VOICE</div>
    <div class="fact-grid">${cards.join("")}</div>
  `;
}

function clockHour(hour: number): number {
  return ((hour + 24) % 12) || 12;
}

function meetingsLine(seconds: number): string {
  return wholeMinutes(seconds) > 0
    ? `captured this week — ${formatCoarseDuration(seconds)} of talk turned into notes you can search.`
    : "captured this week, turned into notes you can search.";
}

function fact(name: IconName, blue: boolean, title: string, line: string): string {
  return `
    <div class="fact-card">
      <span class="fact-card-icon${blue ? " fact-card-icon--blue" : ""}">${icon(name)}</span>
      <div>${title}<span>${line}</span></div>
    </div>
  `;
}

function count(value: number, noun: string): string {
  return `${value.toLocaleString("en-US")} ${noun}${value === 1 ? "" : "s"}`;
}
