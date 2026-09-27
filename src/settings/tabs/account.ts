/**
 * The Account panel: who uses JustSay on this computer, by the name the OS
 * knows them by. Sign-in will add the photo and email to the same card.
 */
import { initialsOf } from "../shell/account-name";

export function renderAccount(container: HTMLElement, name: string): void {
  container.innerHTML = `
    <h2 class="panel-title">Account</h2>
    <p class="panel-subtitle">On this computer.</p>
    <div class="card account-card">
      <span class="avatar avatar--large"></span>
      <div class="account-card-name"></div>
    </div>
  `;
  container.querySelector(".avatar")!.textContent = initialsOf(name);
  container.querySelector(".account-card-name")!.textContent = name;
}
