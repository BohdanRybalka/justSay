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
  bindIndicatorActivation,
} from "../../status-indicator";

let prevLastError: string | null = null;

export function isStaleStatusResponse(requestToken: number, latestIssuedToken: number): boolean {
  return requestToken !== latestIssuedToken;
}

export function renderModels(container: HTMLElement, settings: UserSettings): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Models</h2>

    <div class="setting-group">
      <div class="setting-label">Speech-to-Text</div>
      <div class="setting-row">
        <span class="label">Mode</span>
        <div class="toggle-group">
          <button class="toggle-btn ${settings.stt_mode === "cloud" ? "active" : ""}" id="stt-cloud">Cloud</button>
          <button class="toggle-btn ${settings.stt_mode === "local" ? "active" : ""}" id="stt-local">Local</button>
          <span id="stt-local-indicator" class="status-indicator-badge"></span>
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
  let latestSttStatusToken = 0;

  function renderCurrentStt() {
    if (currentSttMode === "cloud") {
      renderIndicator(sttLocalIndicator, "idle");
      sttPanel.innerHTML = "";
    } else {
      sttPanel.innerHTML = '<div class="setting-hint" id="stt-local-caption"></div>';
      refreshSttStatus();
    }
  }

  function applyLocalIndicator(error: string | null, ready: boolean, captionText: string) {
    const state = computeIndicatorState({ active: currentSttMode === "local", ready, error });
    renderIndicator(sttLocalIndicator, state, {
      title: error ?? "",
      interactive: state === "error",
      ariaLabel: error
        ? `Local speech-to-text error: ${error}. Press Enter or Space to retry.`
        : undefined,
    });
    const caption = sttPanel.querySelector<HTMLElement>("#stt-local-caption");
    if (caption) caption.textContent = captionText;
    if (onIndicatorStateChange(prevLastError, error)) {
      notifyError(error!);
    }
    prevLastError = error;
  }

  async function refreshSttStatus() {
    if (currentSttMode !== "local") return;
    const token = ++latestSttStatusToken;
    try {
      const s: LocalSttStatus = await api.sttLocalStatus();
      if (isStaleStatusResponse(token, latestSttStatusToken) || currentSttMode !== "local") return;
      applyLocalIndicator(s.last_error, s.model_loaded, `${s.model_name} · ${s.device}`);
    } catch {
      if (isStaleStatusResponse(token, latestSttStatusToken) || currentSttMode !== "local") return;
      applyLocalIndicator("Backend not responding", false, "Backend not responding");
    }
  }

  bindIndicatorActivation(sttLocalIndicator, () => {
    (async () => {
      try {
        await api.sttLocalPrewarm();
      } catch {  }
      await refreshSttStatus();
    })();
  });

  async function switchStt(mode: "cloud" | "local") {
    if (currentSttMode === mode) return;
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

  renderCurrentStt();

  const pollInterval = setInterval(() => {
    if (currentSttMode === "local") refreshSttStatus();
  }, 3000);

  return () => {
    clearInterval(pollInterval);
  };
}
