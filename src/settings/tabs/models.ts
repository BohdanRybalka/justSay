import {
  api,
  type UserSettings,
  type LocalSttStatus,
} from "../../api";
import { loadSettings } from "../settings";
import { notifyError } from "../../notify";
import {
  computeIndicatorState,
  onIndicatorStateChange,
  renderIndicator,
} from "../../status-indicator";

// Edge-triggered latch for the Local STT indicator's last-seen error — drives
// onIndicatorStateChange() so notifyError() fires once per new error, not
// once per 3-second poll while the same error persists.
let prevLastError: string | null = null;

export function renderModels(container: HTMLElement, settings: UserSettings): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Models</h2>

    <div class="setting-group">
      <div class="setting-label">Speech-to-Text</div>
      <div class="setting-row">
        <span class="label">Mode</span>
        <div class="toggle-group">
          <button class="toggle-btn ${settings.stt_mode === "cloud" ? "active" : ""}" id="stt-cloud">Cloud</button>
          <button class="toggle-btn ${settings.stt_mode === "local" ? "active" : ""}" id="stt-local">Local<span id="stt-local-indicator" class="status-indicator-badge"></span></button>
        </div>
      </div>
      <div class="setting-row" id="stt-engine-row" style="${settings.stt_mode === "cloud" ? "" : "display:none;"}">
        <span class="label">Cloud engine
          <span class="info-tip" title="Auto: short clips (≤ 30 s) go to Groq Whisper for speed; long audio and AI Prompt go to Gemini Native Audio. Pin Groq or Gemini to force one provider — pinned-Groq automatically falls back to Gemini for AI Prompt and unsupported formats (.webm).">&#9432;</span>
        </span>
        <select id="stt-engine">
          <option value="auto" ${settings.stt_engine === "auto" ? "selected" : ""}>Auto (recommended)</option>
          <option value="groq" ${settings.stt_engine === "groq" ? "selected" : ""}>Groq Whisper (fast, short)</option>
          <option value="gemini" ${settings.stt_engine === "gemini" ? "selected" : ""}>Gemini (long / structured)</option>
        </select>
      </div>
      <div class="setting-hint">
        Cloud short (&le; 30 s) → Groq Whisper · Cloud long / AI Prompt → Gemini · Local → faster-whisper (NVIDIA/CPU) or whisper.cpp+Vulkan (Windows AMD/Intel)
      </div>
      <div id="stt-panel"></div>
    </div>
  `;

  const sttCloud = container.querySelector<HTMLButtonElement>("#stt-cloud")!;
  const sttLocal = container.querySelector<HTMLButtonElement>("#stt-local")!;
  const sttLocalIndicator = container.querySelector<HTMLElement>("#stt-local-indicator")!;
  const sttPanel = container.querySelector<HTMLElement>("#stt-panel")!;
  const engineRow = container.querySelector<HTMLElement>("#stt-engine-row")!;
  const engineSelect = container.querySelector<HTMLSelectElement>("#stt-engine")!;

  engineSelect.addEventListener("change", async () => {
    const value = engineSelect.value as "auto" | "groq" | "gemini";
    await api.updateSettings({ stt_engine: value });
    await loadSettings();
  });

  let currentSttMode = settings.stt_mode;

  // --- Render panels based on current mode ---
  function renderCurrentStt() {
    if (currentSttMode === "cloud") {
      renderIndicator(sttLocalIndicator, "idle");
      sttPanel.innerHTML = "";
    } else {
      sttPanel.innerHTML = '<div class="setting-hint" id="stt-local-caption"></div>';
      refreshSttStatus();
    }
  }

  // --- STT indicator + caption update, shared by the success/failure paths ---
  function applyLocalIndicator(error: string | null, ready: boolean, captionText: string) {
    const state = computeIndicatorState({ active: currentSttMode === "local", ready, error });
    renderIndicator(sttLocalIndicator, state, { title: error ?? "" });
    const caption = sttPanel.querySelector<HTMLElement>("#stt-local-caption");
    if (caption) caption.textContent = captionText;
    if (onIndicatorStateChange(prevLastError, error)) {
      notifyError(error!);
    }
    prevLastError = error;
  }

  // --- STT Status refresh ---
  async function refreshSttStatus() {
    if (currentSttMode !== "local") return;
    try {
      const s: LocalSttStatus = await api.sttLocalStatus();
      // Re-check after the await: the user may have switched to Cloud while
      // this request was in flight (e.g. a slow first-run pip-install poll
      // tick). A stale response must not touch the badge/caption/latch or
      // fire notifyError() — Cloud mode must never surface a Local STT toast.
      if (currentSttMode !== "local") return;
      applyLocalIndicator(s.last_error, s.model_loaded, `${s.model_name} · ${s.device}`);
    } catch {
      if (currentSttMode !== "local") return;
      applyLocalIndicator("Backend not responding", false, "Backend not responding");
    }
  }

  // --- Retry-on-click when the indicator shows an error ---
  sttLocalIndicator.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!sttLocalIndicator.classList.contains("status-indicator-badge--error")) return;
    (async () => {
      try {
        await api.sttLocalPrewarm();
      } catch { /* surfaced via the next status poll */ }
      await refreshSttStatus();
    })();
  });

  // --- Mode toggle handler ---
  async function switchStt(mode: "cloud" | "local") {
    if (currentSttMode === mode) return;
    // No explicit unload here: PUT /stt/mode's clear_cache() already tears
    // down the Local provider unconditionally on every mode change, so a
    // frontend pre-emptive unload was always redundant.
    currentSttMode = mode;
    sttCloud.classList.toggle("active", mode === "cloud");
    sttLocal.classList.toggle("active", mode === "local");
    engineRow.style.display = mode === "cloud" ? "" : "none";
    await api.setSttMode(mode);
    await loadSettings();
    renderCurrentStt();
  }

  sttCloud.addEventListener("click", () => switchStt("cloud"));
  sttLocal.addEventListener("click", () => switchStt("local"));

  // --- Initial render ---
  renderCurrentStt();

  // --- Polling ---
  const pollInterval = setInterval(() => {
    if (currentSttMode === "local") refreshSttStatus();
  }, 3000);

  return () => {
    clearInterval(pollInterval);
  };
}
