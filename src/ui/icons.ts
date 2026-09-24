/**
 * The design's icon set: 24x24 stroked symbols mounted once per page as an SVG
 * sprite, referenced by name through `icon()`.
 */

const ICON_SYMBOLS = {
  mic: '<path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 18v4"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  alert: '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  sliders: '<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3"/><path d="M14 2v4M8 10v4M16 18v4"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  chart: '<path d="M3 21h18"/><path d="M6 21v-6M12 21V8M18 21v-9"/>',
  cog: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20"/>',
  file: '<path d="M14.5 2h-8a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7.5Z"/><path d="M14.5 2v5.5h5"/><path d="M8.5 15v2M11.5 13v6M14.5 15v2"/>',
  upload: '<path d="M12 16V4"/><path d="m7.5 8.5 4.5-4.5 4.5 4.5"/><path d="M4 15v3a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-3"/>',
  search: '<circle cx="11" cy="11" r="7.5"/><path d="m21 21-4.5-4.5"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  star: '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.2-5.4-2.9-5.4 2.9 1-6.2L3.2 9.5l6.1-.9Z"/>',
  dots: '<circle cx="5" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="19" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
  flame: '<path d="M12 22c4 0 7-2.8 7-6.6 0-4.4-4.2-6-5.4-10.4C11.4 7 10 8.4 10 10.5c0 1.4.6 2.3.6 2.3S9 12 8.4 9.9C6.6 11.6 5 13.4 5 15.4 5 19.2 8 22 12 22Z"/>',
  book: '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v18H6.5A2.5 2.5 0 0 0 4 22.5Z"/><path d="M4 17.5A2.5 2.5 0 0 1 6.5 15H20"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.5 1.5M17.6 17.6l1.5 1.5M2 12h2M20 12h2M4.9 19.1l1.5-1.5M17.6 6.4l1.5-1.5"/>',
  moon: '<path d="M12 3a6.5 6.5 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  chev: '<path d="m6 9 6 6 6-6"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  min: '<path d="M5 12h14"/>',
  sq: '<rect x="5" y="5" width="14" height="14" rx="2.5"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  refresh: '<path d="M3 12a9 9 0 0 1 15.3-6.4L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.3 6.4L3 16"/><path d="M3 21v-5h5"/>',
  share: '<path d="M12 3v13"/><path d="m7.5 7.5 4.5-4.5 4.5 4.5"/><path d="M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>',
  folder: '<path d="M20 20a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-7.5l-2-2.6A2 2 0 0 0 8.9 3.5H4a2 2 0 0 0-2 2V18a2 2 0 0 0 2 2Z"/>',
  cloud: '<path d="M17.5 19H9a7 7 0 1 1 6.7-9h1.8a4.5 4.5 0 1 1 0 9Z"/>',
  chip: '<rect x="4.5" y="4.5" width="15" height="15" rx="2.5"/><rect x="9" y="9" width="6" height="6" rx="1.2"/><path d="M9 2v2.5M15 2v2.5M9 19.5V22M15 19.5V22M2 9h2.5M2 15h2.5M19.5 9H22M19.5 15H22"/>',
  users: '<path d="M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="3.4"/><path d="M22 20v-2a4 4 0 0 0-3-3.8"/><path d="M16 3.6a4 4 0 0 1 0 6.8"/>',
  speaker: '<path d="M11 5 6.5 9H3v6h3.5L11 19Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trend: '<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  sparkle: '<path d="M12 3.5 13.9 9.6 20 11.5 13.9 13.4 12 19.5 10.1 13.4 4 11.5 10.1 9.6Z"/>',
  exit: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/>',
} as const;

export type IconName = keyof typeof ICON_SYMBOLS;
type IconSize = "small" | "regular" | "large";

const SPRITE_ID = "icon-sprite";

export function mountIconSprite(doc: Document): void {
  if (doc.getElementById(SPRITE_ID)) return;
  const symbols = Object.entries(ICON_SYMBOLS)
    .map(([name, body]) => `<symbol id="${name}" viewBox="0 0 24 24">${body}</symbol>`)
    .join("");
  doc.body.insertAdjacentHTML(
    "afterbegin",
    `<svg id="${SPRITE_ID}" width="0" height="0" style="position:absolute" aria-hidden="true"><defs>${symbols}</defs></svg>`,
  );
}

export function icon(name: IconName, size: IconSize = "regular"): string {
  const sizeClass = size === "regular" ? "icon" : `icon icon--${size}`;
  return `<svg class="${sizeClass}" aria-hidden="true"><use href="#${name}"/></svg>`;
}
