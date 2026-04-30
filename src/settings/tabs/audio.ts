import { api, type UserSettings } from "../../api";
import { saveSettings } from "../settings";

let levelInterval: ReturnType<typeof setInterval> | null = null;

export function renderAudio(container: HTMLElement, settings: UserSettings): () => void {
  container.innerHTML = `
    <h2 class="tab-title">Audio</h2>

    <div class="setting-group">
      <div class="setting-label">Microphone Test</div>
      <div class="setting-row" style="flex-direction: column; align-items: stretch; gap: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <span class="label" id="rec-label">Click to test microphone</span>
          <button class="btn btn-primary" id="btn-test-mic">Record</button>
        </div>
        <div class="level-meter">
          <div class="level-meter-fill" id="level-fill"></div>
        </div>
      </div>
    </div>

    <div class="setting-group">
      <div class="setting-label">Settings</div>
      <div class="setting-row">
        <span class="label">Sample rate</span>
        <span class="value">16000 Hz</span>
      </div>
      <div class="setting-row">
        <span class="label">Max recording duration</span>
        <div>
          <input type="number" id="max-duration" value="${settings.max_recording_seconds}" min="10" max="3600" style="width: 80px; text-align: right;" />
          <span class="value" style="margin-left: 4px;">sec</span>
        </div>
      </div>
    </div>
  `;

  const btnTest = container.querySelector<HTMLButtonElement>("#btn-test-mic")!;
  const recLabel = container.querySelector<HTMLElement>("#rec-label")!;
  const levelFill = container.querySelector<HTMLElement>("#level-fill")!;
  const maxDuration = container.querySelector<HTMLInputElement>("#max-duration")!;

  let isRecording = false;

  // Mic test
  btnTest.addEventListener("click", async () => {
    if (isRecording) {
      try {
        await api.audioStop();
      } catch { /* ignore */ }
      isRecording = false;
      btnTest.textContent = "Record";
      recLabel.textContent = "Click to test microphone";
      stopLevelPolling();
      levelFill.style.width = "0%";
    } else {
      // Check if already recording (widget might be using it)
      try {
        const status = await api.audioStatus();
        if (status.is_recording) {
          recLabel.textContent = "Microphone busy (widget recording)";
          return;
        }
      } catch { /* ignore */ }

      try {
        await api.audioStart();
        isRecording = true;
        btnTest.textContent = "Stop";
        recLabel.textContent = "Recording...";
        startLevelPolling(levelFill);
      } catch (e) {
        recLabel.textContent = "Failed to start";
        console.error(e);
      }
    }
  });

  // Max duration
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  maxDuration.addEventListener("input", () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const val = parseInt(maxDuration.value, 10);
      if (val >= 10 && val <= 3600) {
        saveSettings({ max_recording_seconds: val });
      }
    }, 500);
  });

  // Cleanup
  return () => {
    stopLevelPolling();
    if (isRecording) {
      api.audioStop().catch(() => {});
    }
    if (debounceTimer) clearTimeout(debounceTimer);
  };
}

function startLevelPolling(fill: HTMLElement) {
  stopLevelPolling();
  levelInterval = setInterval(async () => {
    try {
      const status = await api.audioStatus();
      // dBFS range: -60 (silent) to 0 (max). Map to 0-100%.
      const pct = Math.max(0, Math.min(100, ((status.level_db + 60) / 60) * 100));
      fill.style.width = `${pct}%`;
    } catch { /* ignore */ }
  }, 100);
}

function stopLevelPolling() {
  if (levelInterval) {
    clearInterval(levelInterval);
    levelInterval = null;
  }
}
