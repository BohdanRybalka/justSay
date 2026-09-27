/**
 * The Account panel: who uses JustSay on this computer, by the name they chose
 * or, until they choose one, the name the OS knows them by. Clicking the name
 * edits it in place. Sign-in will add the photo and email to the same card.
 */
import { DISPLAY_NAME_MAX_LENGTH } from "../../api";
import { notifyError } from "../../notify";
import { displayName, initialsOf } from "../shell/account-name";

export interface AccountName {
  chosen: string;
  osName: string;
}

/** `rename` stores the new chosen name, empty for the OS name, and rejects
 *  when it could not. */
export function renderAccount(
  container: HTMLElement,
  name: AccountName,
  rename: (chosen: string) => Promise<void>,
): void {
  container.innerHTML = `
    <h2 class="panel-title">Account</h2>
    <p class="panel-subtitle">On this computer.</p>
    <div class="card account-card">
      <span class="avatar avatar--large"></span>
      <button type="button" class="account-card-name" title="Change your name"></button>
    </div>
  `;
  const shown = displayName(name.chosen, name.osName);
  container.querySelector(".avatar")!.textContent = initialsOf(shown);
  const nameButton = container.querySelector<HTMLButtonElement>(".account-card-name")!;
  nameButton.textContent = shown || "Add your name";
  nameButton.classList.toggle("account-card-name--empty", !shown);
  nameButton.addEventListener("click", () => editName(container, nameButton, name, shown, rename));
}

function editName(
  container: HTMLElement,
  nameButton: HTMLButtonElement,
  name: AccountName,
  shown: string,
  rename: (chosen: string) => Promise<void>,
): void {
  const input = document.createElement("input");
  input.className = "account-card-name account-card-name-input";
  input.value = shown;
  input.placeholder = name.osName || "Your name";
  input.maxLength = DISPLAY_NAME_MAX_LENGTH;
  input.setAttribute("aria-label", "Your name");
  nameButton.replaceWith(input);
  input.focus();
  input.select();

  let settled = false;
  const finish = async (save: boolean): Promise<void> => {
    if (settled) return;
    settled = true;
    const next = input.value.trim();
    if (!save || next === shown) {
      renderAccount(container, name, rename);
      return;
    }
    try {
      await rename(next);
      renderAccount(container, { ...name, chosen: next }, rename);
    } catch (e) {
      renderAccount(container, name, rename);
      void notifyError(`Could not save your name: ${e instanceof Error ? e.message : String(e)}`);
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") void finish(true);
    else if (event.key === "Escape") void finish(false);
  });
  input.addEventListener("blur", () => void finish(true));
}
