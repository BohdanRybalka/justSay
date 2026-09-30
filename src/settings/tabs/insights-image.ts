/**
 * The clay card of this month as a PNG, drawn with Canvas 2D at twice its size in the
 * bundled fonts, with the card's values from its CSS. The shell copies it to the
 * clipboard or asks where to save it; the bytes cross as a raw body and the page never
 * holds a path.
 */
import { withTimeout } from "../../timeout";

export interface SavedCardView {
  value: string;
  note: string;
  compare: { label: string; fraction: number; time: string; dim: boolean }[];
  figures: { value: string; label: string }[];
}

export interface CardLayout {
  height: number;
  noteLines: string[];
}

const WIDTH = 600;
const SCALE = 2;
const PAD_TOP = 30;
const PAD_SIDE = 32;
const PAD_BOTTOM = 26;
const RADIUS = 18;
const LABEL_LINE = 16.5;
const VALUE_TOP = 12;
const VALUE_LINE = 56;
const NOTE_TOP = 10;
const NOTE_LINE = 21;
const NOTE_MAX_CH = 44;
const COMPARE_TOP = 22;
const COMPARE_GAP = 9;
const COMPARE_ROW = 18;
const COMPARE_LABEL = 72;
const COMPARE_TRACK_MAX = 400;
const COMPARE_TRACK = 14;
const FLEX_GAP = 12;
const FIGURES_TOP = 24;
const FIGURES_PAD = 20;
const FIGURES_GAP = 38;
const FIGURE_LINE = 30;
const FIGURE_LABEL_LINE = 17.25;
const WARM_ANGLE_DEG = 158;
const WARM_FROM = "#C9683A";
const WARM_TO = "#AE4E22";
const COPY_TIMEOUT_MS = 5000;
const SANS = `"Inter", system-ui, sans-serif`;
const MONO = `"JetBrains Mono", ui-monospace, monospace`;
const FONTS = {
  label: `700 11px ${SANS}`,
  value: `700 56px ${MONO}`,
  note: `400 14px ${SANS}`,
  compareLabel: `400 12px ${SANS}`,
  compareTime: `600 12px ${MONO}`,
  figure: `700 20px ${MONO}`,
  figureLabel: `400 11.5px ${SANS}`,
};

type Context = CanvasRenderingContext2D;

export function wrapLines(ctx: Context, text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxWidth) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

export function layoutSavedCard(ctx: Context, view: SavedCardView): CardLayout {
  ctx.font = FONTS.note;
  const noteWidth = Math.min(WIDTH - 2 * PAD_SIDE, ctx.measureText("0").width * NOTE_MAX_CH);
  const noteLines = wrapLines(ctx, view.note, noteWidth);
  const compare = view.compare.length === 0 ? 0 : COMPARE_TOP + view.compare.length * (COMPARE_GAP + COMPARE_ROW);
  const height =
    PAD_TOP + LABEL_LINE + VALUE_TOP + VALUE_LINE + NOTE_TOP + noteLines.length * NOTE_LINE + compare +
    FIGURES_TOP + 1 + FIGURES_PAD + FIGURE_LINE + FIGURE_LABEL_LINE + PAD_BOTTOM;
  return { height: Math.ceil(height), noteLines };
}

export function drawSavedCard(ctx: Context, view: SavedCardView, { height, noteLines }: CardLayout): void {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, 0, WIDTH, height, RADIUS);
  ctx.clip();
  ctx.fillStyle = warmGradient(ctx, height);
  ctx.fillRect(0, 0, WIDTH, height);
  ctx.fillStyle = "rgba(255,255,255,.07)";
  ctx.beginPath();
  ctx.arc(WIDTH - 90, height - 30, 160, 0, Math.PI * 2);
  ctx.fill();

  ctx.textBaseline = "middle";
  let y = PAD_TOP;
  text(ctx, "YOU SAVED THIS MONTH", FONTS.label, 0.8, PAD_SIDE, y + LABEL_LINE / 2, "1.21px");
  y += LABEL_LINE + VALUE_TOP;
  text(ctx, view.value, FONTS.value, 1, PAD_SIDE, y + VALUE_LINE / 2, "-2.52px");
  y += VALUE_LINE + NOTE_TOP;
  for (const line of noteLines) {
    text(ctx, line, FONTS.note, 0.9, PAD_SIDE, y + NOTE_LINE / 2);
    y += NOTE_LINE;
  }

  if (view.compare.length > 0) y += COMPARE_TOP;
  for (const row of view.compare) {
    y += COMPARE_GAP;
    const middle = y + COMPARE_ROW / 2;
    text(ctx, row.label, FONTS.compareLabel, 0.85, PAD_SIDE, middle);
    ctx.font = FONTS.compareTime;
    const timeWidth = ctx.measureText(row.time).width;
    const trackX = PAD_SIDE + COMPARE_LABEL + FLEX_GAP;
    const track = Math.min(COMPARE_TRACK_MAX, WIDTH - PAD_SIDE - trackX - FLEX_GAP - timeWidth);
    const trackY = middle - COMPARE_TRACK / 2;
    pill(ctx, trackX, trackY, track, "rgba(255,255,255,.22)");
    pill(ctx, trackX, trackY, track * row.fraction, row.dim ? "rgba(255,255,255,.4)" : "rgba(255,255,255,.94)");
    text(ctx, row.time, FONTS.compareTime, 1, trackX + track + FLEX_GAP, middle);
    y += COMPARE_ROW;
  }

  y += FIGURES_TOP;
  ctx.fillStyle = "rgba(255,255,255,.22)";
  ctx.fillRect(PAD_SIDE, y, WIDTH - 2 * PAD_SIDE, 1);
  y += 1 + FIGURES_PAD;
  let x = PAD_SIDE;
  for (const figure of view.figures) {
    const valueWidth = text(ctx, figure.value, FONTS.figure, 1, x, y + FIGURE_LINE / 2, "-0.56px");
    const labelWidth = text(ctx, figure.label, FONTS.figureLabel, 0.8, x, y + FIGURE_LINE + FIGURE_LABEL_LINE / 2);
    x += Math.max(valueWidth, labelWidth) + FIGURES_GAP;
  }
  ctx.restore();
}

/** Draws `content` in white at `alpha` and answers its width. */
function text(ctx: Context, content: string, font: string, alpha: number, x: number, y: number, spacing = "0px"): number {
  ctx.font = font;
  ctx.letterSpacing = spacing;
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "#fff";
  ctx.fillText(content, x, y);
  ctx.globalAlpha = 1;
  return ctx.measureText(content).width;
}

function pill(ctx: Context, x: number, y: number, width: number, colour: string): void {
  if (width <= 0) return;
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.roundRect(x, y, width, COMPARE_TRACK, COMPARE_TRACK / 2);
  ctx.fill();
}

/** The card's `linear-gradient(158deg, …)` as the CSS gradient line spans the card. */
function warmGradient(ctx: Context, height: number): CanvasGradient {
  const angle = (WARM_ANGLE_DEG * Math.PI) / 180;
  const [dx, dy] = [Math.sin(angle), -Math.cos(angle)];
  const half = (Math.abs(WIDTH * dx) + Math.abs(height * dy)) / 2;
  const [cx, cy] = [WIDTH / 2, height / 2];
  const gradient = ctx.createLinearGradient(cx - dx * half, cy - dy * half, cx + dx * half, cy + dy * half);
  gradient.addColorStop(0, WARM_FROM);
  gradient.addColorStop(1, WARM_TO);
  return gradient;
}

export async function savedCardPng(view: SavedCardView): Promise<Uint8Array> {
  await Promise.all(Object.values(FONTS).map((font) => document.fonts.load(font)));
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const layout = layoutSavedCard(ctx, view);
  canvas.width = WIDTH * SCALE;
  canvas.height = layout.height * SCALE;
  ctx.scale(SCALE, SCALE);
  drawSavedCard(ctx, view, layout);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (blob === null) throw new Error("The card could not be encoded as PNG");
  return new Uint8Array(await blob.arrayBuffer());
}

async function invokeWithImage<T>(command: string, png: Uint8Array, headers?: Record<string, string>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, png, headers === undefined ? undefined : { headers });
}

/** Puts the image on the clipboard, kept off other devices as copied text is. */
export async function copyImage(png: Uint8Array): Promise<void> {
  await withTimeout(invokeWithImage("copy_image", png), COPY_TIMEOUT_MS);
}

/** Asks where to save the image and writes it there; `false` when the user cancelled. */
export function saveImage(png: Uint8Array, suggestedName: string): Promise<boolean> {
  return invokeWithImage<boolean>("save_image", png, { "x-suggested-name": suggestedName });
}
