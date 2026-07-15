import {
  api,
  sseStream,
  type UserSettings,
  type LocalSttStatus,
} from "../../api";
import { loadSettings } from "../settings";
import { escapeHtml } from "../html";

const WHISPER_MODELS = ["large-v3-turbo", "large-v3", "large-v2", "medium", "small", "base", "tiny"];
const WHISPER_DEVICES = ["auto", "cpu", "cuda"];

// SSE abort controller for STT install stream.
let sttAbort: AbortController | null = null;

// Prevent concurrent operations.
let sttBusy = false;

// Sticky load error — set when /stt/local/load returns 500, cleared when status
// reports model_loaded or last_error becomes null upstream.
let sttLoadError: string | null = null;

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
      sttPanel.innerHTML = `
        <div class="model-info">
          <div class="model-name">gemini-2.5-flash &nbsp;+&nbsp; whisper-large-v3-turbo (Groq)</div>
          <div class="model-detail">Smart-routed: short audio &rarr; Groq · long / structured &rarr; Gemini</div>
        </div>`;
    } else {
      sttPanel.innerHTML = '<div class="local-status"><div class="local-status-row"><span class="ls-label">Loading...</span></div></div>';
      refreshSttStatus();
    }
  }

  // --- STT Status refresh ---
  async function refreshSttStatus() {
    if (currentSttMode !== "local") return;
    try {
      const s = await api.sttLocalStatus();
      renderSttPanel(sttPanel, s);
    } catch {
      sttPanel.innerHTML = '<div class="local-status"><div class="local-status-row"><span class="ls-value" style="color:var(--red)">Backend not responding</span></div></div>';
    }
  }

  function renderSttPanel(panel: HTMLElement, s: LocalSttStatus) {
    if (!s.package_installed) {
      renderSttInstall(panel, s);
      return;
    }

    // Backend's last_error wins; the frontend latch is for in-flight errors
    // that the polling cycle hasn't picked up yet.
    const error = s.last_error || sttLoadError;
    if (s.model_loaded && sttLoadError) {
      // A successful load wipes the local latch.
      sttLoadError = null;
    }

    let statusDot: string, statusText: string, actionHtml: string;
    if (sttBusy) {
      statusDot = "orange";
      statusText = "Loading...";
      actionHtml = "";
    } else if (error && !s.model_loaded) {
      statusDot = "red";
      statusText = "Failed to load";
      actionHtml = '<button class="btn btn-primary btn-sm" id="stt-start">Retry</button>';
    } else if (s.model_loaded) {
      statusDot = "green";
      statusText = `Loaded${s.model_ram_mb ? ` · ${s.model_ram_mb} MB RSS` : ""} · ${s.device}`;
      actionHtml = '<button class="btn btn-secondary btn-sm" id="stt-stop">Stop</button>';
    } else {
      statusDot = "gray";
      statusText = "Stopped";
      actionHtml = '<button class="btn btn-primary btn-sm" id="stt-start">Start</button>';
    }

    // Dropdowns are gated on model_loaded — changing model while loaded would
    // invalidate the cache mid-run. Force the user to Stop first.
    const dropdownDisabled = s.model_loaded || sttBusy ? "disabled" : "";

    const modelOptions = WHISPER_MODELS.map(m =>
      `<option value="${m}" ${m === s.model_name ? "selected" : ""}>${m}</option>`
    ).join("");
    const deviceOptions = WHISPER_DEVICES.map(d =>
      `<option value="${d}" ${d === s.device ? "selected" : ""}>${d}${d === "auto" ? ` (${s.gpu_available ? "CUDA" : "CPU"})` : ""}</option>`
    ).join("");

    const errorBlock = error
      ? `<div class="local-status-error"><span class="ls-label">Error</span><span class="ls-error-text">${escapeHtml(error)}</span></div>`
      : "";

    panel.innerHTML = `
      <div class="local-status">
        <div class="local-status-row">
          <span class="ls-label">Status</span>
          <span class="ls-value"><span class="dot ${statusDot}"></span>${statusText}</span>
          ${actionHtml ? `<span>${actionHtml}</span>` : ""}
        </div>
        <div class="local-status-row">
          <span class="ls-label">Model</span>
          <select id="stt-model-sel" ${dropdownDisabled}>${modelOptions}</select>
        </div>
        <div class="local-status-row">
          <span class="ls-label">Device</span>
          <select id="stt-device-sel" ${dropdownDisabled}>${deviceOptions}</select>
        </div>
        ${s.gpu_available ? `<div class="local-status-row"><span class="ls-label">GPU</span><span class="ls-value">${s.gpu_name}</span></div>` : ""}
        ${errorBlock}
      </div>
    `;

    panel.querySelector("#stt-start")?.addEventListener("click", async () => {
      sttBusy = true;
      sttLoadError = null;
      renderSttPanel(panel, { ...s, model_loaded: false, last_error: null });
      try {
        await api.sttLocalLoad();
      } catch (err) {
        sttLoadError = (err as Error).message || "Unknown error";
      }
      sttBusy = false;
      await refreshSttStatus();
    });

    panel.querySelector("#stt-stop")?.addEventListener("click", async () => {
      sttBusy = true;
      sttLoadError = null;
      renderSttPanel(panel, { ...s, model_loaded: false, last_error: null });
      try {
        await api.sttLocalUnload();
      } catch { /* ignore */ }
      sttBusy = false;
      await refreshSttStatus();
    });

    panel.querySelector("#stt-model-sel")?.addEventListener("change", async (e) => {
      const select = e.target as HTMLSelectElement;
      if (s.model_loaded) {
        // Defensive: dropdown should already be disabled, but if it isn't,
        // bail out without touching the cache.
        select.value = s.model_name;
        return;
      }
      sttLoadError = null;
      await api.updateSettings({ whisper_model_size: select.value });
      await loadSettings();
      await refreshSttStatus();
    });

    panel.querySelector("#stt-device-sel")?.addEventListener("change", async (e) => {
      const select = e.target as HTMLSelectElement;
      if (s.model_loaded) {
        select.value = s.device;
        return;
      }
      sttLoadError = null;
      await api.updateSettings({ whisper_device: select.value });
      await loadSettings();
      await refreshSttStatus();
    });
  }

  function renderSttInstall(panel: HTMLElement, _s: LocalSttStatus) {
    panel.innerHTML = `
      <div class="local-status">
        <div class="local-status-row">
          <span class="ls-label">Status</span>
          <span class="ls-value"><span class="dot red"></span>Package not installed</span>
          <span><button class="btn btn-primary btn-sm" id="stt-install-btn">Install</button></span>
        </div>
        <div id="stt-progress"></div>
        <div id="stt-error" style="font-size:11px;color:var(--red);margin-top:6px"></div>
      </div>
    `;

    panel.querySelector("#stt-install-btn")!.addEventListener("click", () => {
      const btn = panel.querySelector<HTMLButtonElement>("#stt-install-btn")!;
      btn.disabled = true; btn.textContent = "Installing...";
      const progressEl = panel.querySelector<HTMLElement>("#stt-progress")!;
      progressEl.innerHTML = '<div class="progress-text"><span id="stt-progress-status">Installing...</span></div>';

      sttAbort = sseStream("/stt/local/install",
        (d) => { const el = panel.querySelector("#stt-progress-status"); if (el) el.textContent = d.status || "Installing..."; },
        () => { sttAbort = null; refreshSttStatus(); },
        (err) => { sttAbort = null; btn.disabled = false; btn.textContent = "Retry"; panel.querySelector<HTMLElement>("#stt-error")!.textContent = err; },
      );
    });
  }

  // --- Mode toggle handler ---
  async function switchStt(mode: "cloud" | "local") {
    if (currentSttMode === mode) return;
    if (currentSttMode === "local") {
      try { await api.sttLocalUnload(); } catch { /* ignore */ }
    }
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
    if (currentSttMode === "local" && !sttBusy) refreshSttStatus();
    updateResources(resourcesPanel);
  }, 3000);

  return () => {
    clearInterval(pollInterval);
    if (sttAbort) { sttAbort.abort(); sttAbort = null; }
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

    const gpuHtml = r.gpu
      ? renderResourceBar(
          "GPU VRAM",
          r.gpu.name,
          r.gpu.vram_total_mb > 0
            ? Math.min((r.gpu.vram_used_mb / r.gpu.vram_total_mb) * 100, 100)
            : 0,
          `${(r.gpu.vram_used_mb / 1024).toFixed(1)} / ${(r.gpu.vram_total_mb / 1024).toFixed(1)} GB`,
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
