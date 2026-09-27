/**
 * The main window's own 40px title bar. The bar, the mark and the name are the
 * drag region. Windows gets minimise, maximise and close drawn here; macOS
 * draws its native traffic lights over the bar's left end, so there the bar
 * only makes room for them.
 */
import type { Window } from "@tauri-apps/api/window";
import type { ShortcutPlatform } from "../../accelerator";
import { icon } from "../../ui/icons";

type TitlebarWindow = Pick<Window, "minimize" | "toggleMaximize" | "close" | "isMaximized" | "onResized">;

const DRAG_REGION = "data-tauri-drag-region";

export function renderTitlebar(bar: HTMLElement, platform: ShortcutPlatform): void {
  bar.setAttribute(DRAG_REGION, "");
  bar.innerHTML =
    `<span class="brand-mark titlebar-mark" ${DRAG_REGION}>${icon("mic")}</span>` +
    `<span class="titlebar-name" ${DRAG_REGION}>JustSay</span>`;
  if (platform === "mac") {
    bar.classList.add("titlebar--mac");
    return;
  }
  bar.insertAdjacentHTML(
    "beforeend",
    `<div class="titlebar-controls">` +
      `<button type="button" class="titlebar-button" id="btn-minimise" aria-label="Minimise">${icon("min", "small")}</button>` +
      `<button type="button" class="titlebar-button" id="btn-maximise" aria-label="Maximise">${icon("sq", "small")}</button>` +
      `<button type="button" class="titlebar-button titlebar-button--close" id="btn-close" aria-label="Close">${icon("x", "small")}</button>` +
      `</div>`,
  );
}

function renderMaximiseButton(button: HTMLButtonElement, maximised: boolean): void {
  button.innerHTML = icon(maximised ? "restore" : "sq", "small");
  button.setAttribute("aria-label", maximised ? "Restore" : "Maximise");
}

/** Connect the drawn controls to the window. Close goes through the window's
 *  close request, which the shell turns into hiding the window. */
export async function wireTitlebar(bar: HTMLElement, appWindow: TitlebarWindow): Promise<void> {
  const maximise = bar.querySelector<HTMLButtonElement>("#btn-maximise");
  if (!maximise) return;
  bar.querySelector("#btn-minimise")?.addEventListener("click", () => void appWindow.minimize());
  maximise.addEventListener("click", () => void appWindow.toggleMaximize());
  bar.querySelector("#btn-close")?.addEventListener("click", () => void appWindow.close());

  const followMaximised = async () => renderMaximiseButton(maximise, await appWindow.isMaximized());
  await appWindow.onResized(() => void followMaximised());
  await followMaximised();
}
