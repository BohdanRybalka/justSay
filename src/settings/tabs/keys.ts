import { api, type UserSettings } from "../../api";

export function renderKeys(container: HTMLElement, settings: UserSettings): () => void {
  const geminiStored = settings.gemini_api_key === "***";
  const groqStored = settings.groq_api_key === "***";

  container.innerHTML = `
    <h2 class="tab-title">API Keys</h2>
    <div class="setting-hint" style="margin-bottom: 12px;">
      Keys are stored in <code>~/.justsay/settings.json</code>. They are never sent back in API responses.
    </div>

    <div class="setting-group">
      <div class="setting-label">Gemini (STT — long audio &amp; structured)</div>
      <div class="setting-row" style="flex-direction: column; align-items: stretch; gap: 8px;">
        <div class="key-input-row">
          <input type="password" id="gemini-key-input" autocomplete="off" spellcheck="false"
            placeholder="${geminiStored ? "•••••••• (key stored)" : "Paste your Gemini API key"}" />
          <button class="btn btn-secondary btn-sm" id="gemini-reveal" title="Reveal / hide">👁</button>
          <button class="btn btn-primary btn-sm" id="gemini-save" disabled>Save</button>
        </div>
        <div class="setting-hint" id="gemini-status">${geminiStored ? "Key stored." : "No key set — cloud STT will fail."}</div>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Groq (STT — short audio &amp; LLM)</div>
      <div class="setting-row" style="flex-direction: column; align-items: stretch; gap: 8px;">
        <div class="key-input-row">
          <input type="password" id="groq-key-input" autocomplete="off" spellcheck="false"
            placeholder="${groqStored ? "•••••••• (key stored)" : "Paste your Groq API key"}" />
          <button class="btn btn-secondary btn-sm" id="groq-reveal" title="Reveal / hide">👁</button>
          <button class="btn btn-primary btn-sm" id="groq-save" disabled>Save</button>
        </div>
        <div class="setting-hint" id="groq-status">${groqStored ? "Key stored." : "No key set — cloud STT and LLM will fail."}</div>
      </div>
    </div>
  `;

  function wireKey(
    inputId: string,
    revealId: string,
    saveId: string,
    statusId: string,
    field: "gemini_api_key" | "groq_api_key",
  ) {
    const input = container.querySelector<HTMLInputElement>(`#${inputId}`)!;
    const revealBtn = container.querySelector<HTMLButtonElement>(`#${revealId}`)!;
    const saveBtn = container.querySelector<HTMLButtonElement>(`#${saveId}`)!;
    const status = container.querySelector<HTMLElement>(`#${statusId}`)!;

    input.addEventListener("input", () => {
      saveBtn.disabled = input.value.trim() === "";
    });

    revealBtn.addEventListener("click", () => {
      input.type = input.type === "password" ? "text" : "password";
    });

    saveBtn.addEventListener("click", async () => {
      const value = input.value.trim();
      if (!value) return;

      saveBtn.disabled = true;
      saveBtn.textContent = "Saving…";
      status.textContent = "";

      try {
        await api.updateSettings({ [field]: value } as Partial<UserSettings>);
        input.value = "";
        input.placeholder = "•••••••• (key stored)";
        status.textContent = "Key stored.";
        saveBtn.textContent = "Save";
      } catch (err) {
        status.textContent = `Error: ${(err as Error).message}`;
        saveBtn.disabled = false;
        saveBtn.textContent = "Save";
      }
    });

    // Activate Save on any non-empty input
    input.dispatchEvent(new Event("input"));
  }

  wireKey("gemini-key-input", "gemini-reveal", "gemini-save", "gemini-status", "gemini_api_key");
  wireKey("groq-key-input", "groq-reveal", "groq-save", "groq-status", "groq_api_key");

  return () => {};
}
