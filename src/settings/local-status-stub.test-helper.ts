import type { LocalSTTStatus, ModelDownload } from "../api";

/** An engine that reports its download and has its model: the whisper.cpp case, installed. */
export function buildLocalStatus(overrides: Partial<LocalSTTStatus> = {}): LocalSTTStatus {
  return {
    available: true,
    package_installed: true,
    model_downloaded: true,
    model_bytes: 1_624_555_275,
    model_download_reported: true,
    model_loaded: false,
    model_name: "large-v3-turbo",
    model_ram_mb: null,
    gpu_available: true,
    gpu_name: "AMD Radeon RX 5700 XT",
    gpu_vendor: "amd",
    device: "vulkan",
    compute_type: "float16",
    last_error: null,
    ...overrides,
  };
}

export function buildDownload(overrides: Partial<ModelDownload> = {}): ModelDownload {
  return { stage: "idle", bytes_done: 0, bytes_total: null, error: null, failed_step: null, ...overrides };
}

/** What each of the four install steps under a Local row shows: done, running, failed or pending. */
export function installStepStates(area: Element): string[] {
  return Array.from(area.querySelectorAll(".install-step")).map((item) => {
    for (const state of ["done", "running", "failed"]) {
      if (item.classList.contains(`install-step--${state}`)) return state;
    }
    return "pending";
  });
}
