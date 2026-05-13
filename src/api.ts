/**
 * HTTP client for JustSay Python backend.
 */

const BASE_URL = "http://127.0.0.1:9377";

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const opts: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(`${BASE_URL}${path}`, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// --- Types ---

export interface HealthResponse {
  status: string;
  version: string;
  stt_mode: "cloud" | "local";
  llm_mode: "cloud" | "local";
}

export interface RecordingStatus {
  is_recording: boolean;
  duration_seconds: number;
  level_db: number;
}

export interface DictateResponse {
  text: string;
  duration_ms: number;
  copied_to_clipboard: boolean;
  model_name?: string;
  fallback_reason?: string | null;
}

export interface UserSettings {
  language: string;
  shortcut: string;
  output_dir: string;
  stt_mode: "cloud" | "local";
  llm_mode: "cloud" | "local";
  stt_engine: "auto" | "groq" | "gemini";
  whisper_model_size: string;
  whisper_device: string;
  ollama_host: string;
  ollama_model: string;
  max_recording_seconds: number;
  transcription_style: "normal" | "ai_prompt";
  /** Audio duration (seconds) at or below which the pipeline picks Groq Whisper
   *  in CLOUD mode. Above the threshold (or for `ai_prompt` style) it routes
   *  to Gemini. */
  cloud_routing_threshold: number;
  /** Custom vocabulary / glossary. Plumbed into every STT provider — see the
   *  Python `STTSettings.initial_prompt` docstring for per-provider semantics.
   *  Backend enforces a 500-char ceiling. */
  initial_prompt: string;
  /** Cloud API keys. Always returned as `"***"` (set) or `""` (not set) by GET/PUT.
   *  Send the real key to set it; sending `"***"` is a no-op (backend ignores it). */
  gemini_api_key: string;
  groq_api_key: string;
}

export interface CloudKeyStatus {
  gemini_key_set: boolean;
  groq_key_set: boolean;
}

export interface LocalSttStatus {
  package_installed: boolean;
  model_loaded: boolean;
  model_name: string;
  model_ram_mb: number | null;
  gpu_available: boolean;
  gpu_name: string | null;
  device: string;
  compute_type: string;
  last_error: string | null;
}

export interface OllamaModel {
  name: string;
  size_bytes: number | null;
  parameter_size: string | null;
}

export interface LocalLlmStatus {
  ollama_running: boolean;
  ollama_version: string | null;
  model_downloaded: boolean;
  model_name: string;
  model_size_bytes: number | null;
  model_loaded: boolean;
  vram_used_bytes: number | null;
  available_models: OllamaModel[];
}

export interface GpuInfo {
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
}

export interface ResourceInfo {
  cpu_cores: number;
  cpu_threads: number;
  cpu_percent_total: number;
  cpu_percent_process: number;
  ram_total_mb: number;
  ram_used_mb: number;
  ram_available_mb: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_available_gb: number;
  pid_ram_mb: number;
  pid_ram_gb: number;
  gpu: GpuInfo | null;
}

export interface HistoryStats {
  total_entries: number;
  total_words: number;
  total_audio_seconds: number;
  today_words: number;
  week_words: number;
  by_language: Record<string, number>;
  by_model: Record<string, number>;
}

export interface StorageInfo {
  temp_dir: string;
  temp_size_bytes: number;
  output_dir: string;
  history_path: string;
  history_entries: number;
}

export interface SettingsUpdateResponse {
  settings: UserSettings;
  warning: string | null;
}

export interface CleanupResult {
  freed_bytes: number;
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  language: string;
  style: string;
  text: string;
  duration_ms: number;
  model_name: string | null;
  tokens_used: number | null;
  audio_duration_seconds: number | null;
  word_count: number | null;
  /** Populated only by /history/search responses. Already HTML-escaped on
   *  the backend with `<mark>…</mark>` wrappers around matched spans —
   *  assign directly to `innerHTML`, do NOT re-escape. */
  highlighted_text?: string;
}

export interface HistoryListResponse {
  entries: HistoryEntry[];
  total: number;
}

export interface WordCount {
  word: string;
  count: number;
}

export interface TopWordsResponse {
  items: WordCount[];
  scanned: number;
}

export interface InsightsResponse {
  model: string;
  insights: string[];
  scanned_words: number;
}

// --- API ---

export const api = {
  health: () => request<HealthResponse>("GET", "/health"),

  audioStart: () => request<RecordingStatus>("POST", "/audio/start"),

  audioStop: () => request<{ filename: string; duration_seconds: number }>("POST", "/audio/stop"),

  audioStatus: () => request<RecordingStatus>("GET", "/audio/status"),

  dictate: (language = "uk", style = "normal") =>
    request<DictateResponse>("POST", `/pipeline/dictate?language=${language}&style=${style}`),

  /** Upload an audio file to the pipeline. Accepts an ArrayBuffer of file bytes. */
  processFile: async (
    fileBytes: ArrayBuffer,
    filename: string,
    language = "uk",
    style = "normal",
  ): Promise<DictateResponse> => {
    const form = new FormData();
    const blob = new Blob([fileBytes], { type: "application/octet-stream" });
    form.append("file", blob, filename);
    const url = `${BASE_URL}/pipeline/process-file?language=${language}&style=${style}`;
    const resp = await fetch(url, { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  },

  setSttMode: (mode: "cloud" | "local") =>
    request("PUT", "/stt/mode", { mode }),

  setLlmMode: (mode: "cloud" | "local") =>
    request("PUT", "/llm/mode", { mode }),

  // Resources
  resources: () => request<ResourceInfo>("GET", "/resources"),

  // Local mode status & control
  sttLocalStatus: () => request<LocalSttStatus>("GET", "/stt/local/status"),
  sttLocalLoad: () => request<{ loaded: boolean; model?: string }>("POST", "/stt/local/load"),
  sttLocalUnload: () => request<{ unloaded: boolean }>("POST", "/stt/local/unload"),

  llmLocalStatus: () => request<LocalLlmStatus>("GET", "/llm/local/status"),
  llmLocalLoad: () => request<{ loaded: boolean; error: string | null }>("POST", "/llm/local/load"),
  llmLocalUnload: () => request<{ unloaded: boolean; error: string | null }>("POST", "/llm/local/unload"),
  llmLocalStart: () => request<{ started: boolean; error: string | null }>("POST", "/llm/local/start"),

  // Settings
  getSettings: () => request<UserSettings>("GET", "/settings"),

  updateSettings: (updates: Partial<UserSettings>) =>
    request<SettingsUpdateResponse>("PUT", "/settings", updates),

  getStorageInfo: () => request<StorageInfo>("GET", "/settings/storage"),

  cleanupTemp: () => request<CleanupResult>("POST", "/settings/cleanup"),

  cloudKeyStatus: () => request<CloudKeyStatus>("GET", "/settings/cloud-status"),

  // History
  getHistory: (limit = 50, offset = 0) =>
    request<HistoryListResponse>("GET", `/history?limit=${limit}&offset=${offset}`),

  historyStats: () => request<HistoryStats>("GET", "/history/stats"),

  deleteHistoryEntry: (id: string) =>
    request<{ deleted: boolean }>("DELETE", `/history/${id}`),

  clearHistory: () =>
    request<{ deleted: number }>("DELETE", "/history"),

  searchHistory: (q: string, limit = 30) =>
    request<HistoryListResponse>(
      "GET",
      `/history/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  // Words (Phase 1 — Plan 013)
  wordsTop: (lang: "all" | "uk" | "en" = "all", limit = 50) =>
    request<TopWordsResponse>("GET", `/words/top?lang=${lang}&limit=${limit}`),

  wordsInsights: () => request<InsightsResponse>("GET", "/words/insights"),
};

// --- SSE helpers for model download/pull ---

export interface SSEProgress {
  status: string;
  completed?: number | null;
  total?: number | null;
  error?: string;
  path?: string;
}

export function sseStream(
  path: string,
  onProgress: (data: SSEProgress) => void,
  onDone: (data: SSEProgress) => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE_URL}${path}`, {
    method: "POST",
    signal: controller.signal,
  })
    .then(async (resp) => {
      if (!resp.ok || !resp.body) {
        onError(`HTTP ${resp.status}`);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            try {
              const data: SSEProgress = JSON.parse(line.slice(6));
              if (currentEvent === "done") {
                onDone(data);
              } else if (currentEvent === "error") {
                onError(data.error || "Unknown error");
              } else {
                onProgress(data);
              }
            } catch {
              // skip malformed JSON
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError(String(err));
      }
    });

  return controller;
}
