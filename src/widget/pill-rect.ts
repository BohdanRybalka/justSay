/** The pill's rectangle in physical pixels from the window's top-left corner,
 *  the unit the shell's cursor loop compares the pointer in. */
export interface PillRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** CSS pixels times `pixelRatio`, which carries a WebView2 text-size zoom as
 *  well as the display scale, so the shell tests against what is on screen. */
export function physicalPillRect(box: DOMRectReadOnly, pixelRatio: number): PillRect {
  return {
    x: Math.round(box.left * pixelRatio),
    y: Math.round(box.top * pixelRatio),
    width: Math.round(box.width * pixelRatio),
    height: Math.round(box.height * pixelRatio),
  };
}

/** Hands `report` the pill's rectangle now, and again whenever its size or the
 *  pixel ratio changes — the two things that move it inside a fixed window. */
export function watchPillRect(pill: Element, report: (rect: PillRect) => void): void {
  const send = () =>
    report(physicalPillRect(pill.getBoundingClientRect(), window.devicePixelRatio));
  new ResizeObserver(send).observe(pill);

  const followPixelRatio = () =>
    window
      .matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
      .addEventListener(
        "change",
        () => {
          send();
          followPixelRatio();
        },
        { once: true },
      );
  followPixelRatio();
}
