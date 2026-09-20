import {
  api,
  type UserSettings,
  type LocalSTTStatus,
} from "../../api";
import { loadSettings, type TabLifecycle } from "../settings";
import { displayableError, notifyError } from "../../notify";
import { isStaleStatusResponse } from "../../stale-response";
import {
  computeIndicatorState,
  onIndicatorStateChange,
  renderIndicator,
  bindIndicatorActivation,
} from "../../status-indicator";

let prevLastError: string | null = null;

export function renderModels(container: HTMLElement, settings: UserSettings): TabLifecycle {
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
          <span class="info-tip" title="Auto: short clips (≤ 30 s) go to Groq Whisper for speed; long audio goes to Gemini Native Audio. Pin Groq or Gemini to force one provider — pinned-Groq automatically falls back to Gemini for unsupported formats (.webm).">&#9432;</span>
        </span>
        <select id="stt-engine">
          <option value="auto" ${settings.stt_engine === "auto" ? "selected" : ""}>Auto (recommended)</option>
          <option value="groq" ${settings.stt_engine === "groq" ? "selected" : ""}>Groq Whisper (fast, short)</option>
          <option value="gemini" ${settings.stt_engine === "gemini" ? "selected" : ""}>Gemini (long / any format)</option>
        </select>
      </div>
      <div class="setting-hint">
        Cloud short (&le; 30 s) → Groq Whisper · Cloud long → Gemini · Local → faster-whisper (NVIDIA/CPU) or whisper.cpp+Vulkan (Windows AMD/Intel)
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

  function applyLocalIndicator(
    raw: string | null | undefined,
    ready: boolean,
    captionText: string,
  ) {
    const reported = raw ?? null;
    const error = displayableError(reported);
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
    if (onIndicatorStateChange(prevLastError, reported)) {
      notifyError(error!);
    }
    prevLastError = reported;
  }

  async function refreshSttStatus() {
    if (currentSttMode !== "local") return;
    const token = ++latestSttStatusToken;
    try {
      const s: LocalSTTStatus = await api.sttLocalStatus();
      if (isStaleStatusResponse(token, latestSttStatusToken) || currentSttMode !== "local") return;
      const caption = [s.model_name, s.device].filter(Boolean).join(" · ") || "Local engine";
      applyLocalIndicator(s.last_error, s.model_loaded, caption);
    } catch {
      if (isStaleStatusResponse(token, latestSttStatusToken) || currentSttMode !== "local") return;
      applyLocalIndicator("Backend not responding", false, "Backend not responding");
    }
  }

  bindIndicatorActivation(sttLocalIndicator, () => {
    (async () => {
      try {
        await api.sttLocalPrewarm();
      } catch {}
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

  let pollInterval: ReturnType<typeof setInterval> | null = null;

  /** Start the 3 s status poll, or leave the running one alone — a second
   *  interval over the same handle would be unstoppable. */
  function startSttPolling() {
    if (pollInterval !== null) return;
    pollInterval = setInterval(() => {
      if (currentSttMode === "local") refreshSttStatus();
    }, 3000);
  }

  /** Stop reading the local engine while the window is gone, and forget the
   *  failure the badge was last drawn from — a still-broken engine is worth
   *  announcing once more to a user who has not seen this window since. */
  function releaseResources() {
    if (pollInterval !== null) clearInterval(pollInterval);
    pollInterval = null;
    latestSttStatusToken += 1;
    prevLastError = null;
  }

  /** Restart the interval and read once in this same tick, so the badge a
   *  returning user reads is one request old rather than one interval old.
   *  `refreshSttStatus` mints its own token, and the release bumped the counter
   *  past anything still in flight. */
  function resumeResources() {
    startSttPolling();
    void refreshSttStatus();
  }

  renderCurrentStt();
  startSttPolling();

  return {
    destroy: releaseResources,
    releaseResources,
    resumeResources,
  } satisfies TabLifecycle;
}
