import {
  api,
  sseStream,
  type UserSettings,
  type LocalSttStatus,
} from "../../api";
import { loadSettings } from "../settings";

const WHISPER_MODELS = ["large-v3-turbo", "large-v3", "large-v2", "medium", "small", "base", "tiny"];
const WHISPER_DEVICES = ["auto", "cpu", "cuda"];

// SSE abort controller for STT install stream.
let sttAbort: AbortController | null = null;

// Prevent concurrent operations.
let sttBusy = false;

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
      <div class="setting-hint">
        Cloud short (&le; 30s) → Groq Whisper · Cloud long / AI prompt → Gemini · Local → faster-whisper
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

    let statusDot: string, statusText: string, actionHtml: string;
    if (sttBusy) {
      statusDot = "orange"; statusText = "Loading..."; actionHtml = "";
    } else if (s.model_loaded) {
      statusDot = "green";
      statusText = `Running${s.model_ram_mb ? ` (${s.model_ram_mb} MB)` : ""}`;
      actionHtml = '<button class="btn btn-secondary btn-sm" id="stt-stop">Stop</button>';
    } else {
      statusDot = "gray";
      statusText = "Stopped";
      actionHtml = '<button class="btn btn-primary btn-sm" id="stt-start">Start</button>';
    }

    const dropdownDisabled = s.model_loaded || sttBusy ? "disabled" : "";

    const modelOptions = WHISPER_MODELS.map(m =>
      `<option value="${m}" ${m === s.model_name ? "selected" : ""}>${m}</option>`
    ).join("");
    const deviceOptions = WHISPER_DEVICES.map(d =>
      `<option value="${d}" ${d === s.device ? "selected" : ""}>${d}${d === "auto" ? ` (${s.gpu_available ? "CUDA" : "CPU"})` : ""}</option>`
    ).join("");

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
      </div>
    `;

    panel.querySelector("#stt-start")?.addEventListener("click", async () => {
      sttBusy = true;
      renderSttPanel(panel, { ...s, model_loaded: false });
      try {
        await api.sttLocalLoad();
      } catch { /* status poll will show real state */ }
      sttBusy = false;
      await refreshSttStatus();
    });

    panel.querySelector("#stt-stop")?.addEventListener("click", async () => {
      sttBusy = true;
      renderSttPanel(panel, { ...s, model_loaded: false });
      try {
        await api.sttLocalUnload();
      } catch { /* ignore */ }
      sttBusy = false;
      await refreshSttStatus();
    });

    panel.querySelector("#stt-model-sel")?.addEventListener("change", async (e) => {
      const value = (e.target as HTMLSelectElement).value;
      await api.updateSettings({ whisper_model_size: value });
      await loadSettings();
      await refreshSttStatus();
    });

    panel.querySelector("#stt-device-sel")?.addEventListener("change", async (e) => {
      const value = (e.target as HTMLSelectElement).value;
      await api.updateSettings({ whisper_device: value });
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
    const gpuHtml = r.gpu ? `
      <div class="local-status-row"><span class="ls-label">GPU</span><span class="ls-value">${r.gpu.name}</span></div>
      <div class="local-status-row"><span class="ls-label">VRAM</span><span class="ls-value">${r.gpu.vram_used_mb} / ${r.gpu.vram_total_mb} MB (${r.gpu.vram_free_mb} MB free)</span></div>
    ` : "";
    panel.innerHTML = `
      <div class="local-status-row"><span class="ls-label">CPU</span><span class="ls-value">${r.cpu_cores} cores / ${r.cpu_threads} threads</span></div>
      <div class="local-status-row"><span class="ls-label">RAM</span><span class="ls-value">${r.ram_used_mb} / ${r.ram_total_mb} MB (${r.ram_available_mb} MB free)</span></div>
      <div class="local-status-row"><span class="ls-label">Backend</span><span class="ls-value">${r.pid_ram_mb} MB</span></div>
      ${gpuHtml}
    `;
  } catch {
    panel.innerHTML = '<div class="local-status-row"><span class="ls-value" style="color:var(--text-dim)">—</span></div>';
  }
}
