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
        Cloud short (&le; 30 s) → Groq Whisper · Cloud long / AI Prompt → Gemini · Local → faster-whisper
      </div>
      <div id="stt-panel"></div>
    </div>

    <div class="setting-group">
      <div class="setting-label">System Resources</div>
      <div id="resources-panel" class="local-status">
        <div class="local-status-row"><span class="ls-label">Loading...</span></div>
      </div>
    </div>
  `;

  const sttCloud = container.querySelector<HTMLButtonElement>("#stt-cloud")!;
  const sttLocal = container.querySelector<HTMLButtonElement>("#stt-local")!;
  const sttLocalIndicator = container.querySelector<HTMLElement>("#stt-local-indicator")!;
  const sttPanel = container.querySelector<HTMLElement>("#stt-panel")!;
  const resourcesPanel = container.querySelector<HTMLElement>("#resources-panel")!;
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
      sttPanel.innerHTML = `
        <div class="model-info">
          <div class="model-name">gemini-2.5-flash &nbsp;+&nbsp; whisper-large-v3-turbo (Groq)</div>
          <div class="model-detail">Smart-routed: short audio &rarr; Groq · long / structured &rarr; Gemini</div>
        </div>`;
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
  updateResources(resourcesPanel);

  // --- Polling ---
  const pollInterval = setInterval(() => {
    if (currentSttMode === "local") refreshSttStatus();
    updateResources(resourcesPanel);
  }, 3000);

  return () => {
    clearInterval(pollInterval);
  };
}

// --- Resource monitoring ---

async function updateResources(panel: HTMLElement): Promise<void> {
  try {
    const r = await api.resources();

    const procRamPct = r.ram_total_gb > 0
      ? Math.min((r.pid_ram_gb / r.ram_total_gb) * 100, 100)
      : 0;
    const sysRamPct = r.ram_total_gb > 0
      ? Math.min((r.ram_used_gb / r.ram_total_gb) * 100, 100)
      : 0;

    // vram_used_mb is null for the Windows-registry AMD/Intel source (no
    // live-usage reading) — show the total only, no usage bar/percent.
    const gpuHtml = r.gpu
      ? r.gpu.vram_used_mb !== null
        ? renderResourceBar(
            "GPU VRAM",
            r.gpu.name,
            r.gpu.vram_total_mb > 0
              ? Math.min((r.gpu.vram_used_mb / r.gpu.vram_total_mb) * 100, 100)
              : 0,
            `${(r.gpu.vram_used_mb / 1024).toFixed(1)} / ${(r.gpu.vram_total_mb / 1024).toFixed(1)} GB`,
          )
        : renderResourceRowNoBar(
            "GPU VRAM",
            r.gpu.name,
            `${(r.gpu.vram_total_mb / 1024).toFixed(1)} GB total`,
          )
      : "";

    panel.innerHTML = `
      ${renderResourceBar(
        "Process CPU",
        `Backend (${r.cpu_threads} threads)`,
        r.cpu_percent_process,
        `${r.cpu_percent_process.toFixed(0)}%`,
      )}
      ${renderResourceBar(
        "Process RAM",
        "Backend",
        procRamPct,
        `${r.pid_ram_gb.toFixed(2)} GB`,
      )}
      ${renderResourceBar(
        "System CPU",
        `${r.cpu_cores} cores / ${r.cpu_threads} threads`,
        r.cpu_percent_total,
        `${r.cpu_percent_total.toFixed(0)}%`,
      )}
      ${renderResourceBar(
        "System RAM",
        `${r.ram_available_gb.toFixed(1)} GB free`,
        sysRamPct,
        `${r.ram_used_gb.toFixed(1)} / ${r.ram_total_gb.toFixed(1)} GB`,
      )}
      ${gpuHtml}
    `;
  } catch {
    panel.innerHTML = '<div class="resource-row"><span class="ls-value" style="color:var(--text-dim)">—</span></div>';
  }
}

function renderResourceBar(label: string, sub: string, percent: number, valueText: string): string {
  const safePct = Math.max(0, Math.min(100, percent));
  const tone = safePct > 85 ? "red" : safePct > 60 ? "orange" : "green";
  return `
    <div class="resource-row">
      <div class="resource-row-head">
        <span class="resource-label">${label}</span>
        <span class="resource-value">${valueText}</span>
      </div>
      <div class="resource-bar"><div class="resource-bar-fill ${tone}" style="width:${safePct.toFixed(1)}%"></div></div>
      <div class="resource-sub">${sub}</div>
    </div>
  `;
}

/** Same row layout as `renderResourceBar` but without the fill bar — used
 *  when only a total is known (no live usage reading to compute a percent). */
function renderResourceRowNoBar(label: string, sub: string, valueText: string): string {
  return `
    <div class="resource-row">
      <div class="resource-row-head">
        <span class="resource-label">${label}</span>
        <span class="resource-value">${valueText}</span>
      </div>
      <div class="resource-sub">${sub}</div>
    </div>
  `;
}
