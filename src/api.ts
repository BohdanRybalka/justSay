/**
 * HTTP client for JustSay Python backend.
 */

import { BACKEND_BASE_URL } from "./contracts";
import { TimedOutError } from "./timeout";

/** Why the per-launch token could not be obtained, retained so the UI can name
 *  the failing layer instead of presenting as a dead window (ADR 028).
 *  `bridge-missing` = Tauri's injected bridge scripts never ran (a synchronous
 *  throw); `invoke-timeout` = the bridge ran but the IPC transport never
 *  answered (an unbounded hang, capped here); `invoke-failed` = the command
 *  itself rejected. The three are distinguishable on purpose: they point at
 *  different layers, and on macOS there is nothing else to attach to. */
export type BridgeDiagnosis =
  | { kind: "ok" }
  | { kind: "bridge-missing" }
  | { kind: "invoke-timeout" }
  | { kind: "invoke-failed"; detail: string };

/** Thrown on a `401` so callers can tell "the backend refused this request"
 *  from any other failure, and carry the bridge diagnosis that explains it. */
export class ApiAuthError extends Error {
  readonly diagnosis: BridgeDiagnosis;

  constructor(message: string, diagnosis: BridgeDiagnosis) {
    super(message);
    this.name = "ApiAuthError";
    this.diagnosis = diagnosis;
  }
}

const TOKEN_TIMEOUT_MS = 3000;
const TOKEN_TIMED_OUT = Symbol("token-timed-out");
/** How long a single unanswered `get_backend_token` call may keep being reused
 *  before a fresh one is started anyway. Reuse alone would turn a *recoverable*
 *  wedge — one dropped response on an otherwise live transport — into a
 *  permanent one, because a call that never settles would be raced forever and
 *  no new `invoke()` would ever be attempted. */
export const TOKEN_CALL_REUSE_MS = 60_000;

/** How long an ordinary request may go unanswered before it is abandoned.
 *
 *  `fetch` has no timeout of its own, so a request the backend accepts and then
 *  abandons never settles — and neither does anything sequenced behind it. That
 *  is not a slow window, it is a dead one: the widget's intent queue keeps
 *  handing every later press the same pending promise while the microphone
 *  stays open, and the Settings window never reaches the failure screen its own
 *  40 s bound was added to guarantee.
 *
 *  15 s rather than a rounder number: the slowest control-plane call this app
 *  makes is starting a meeting recording, measured in spec 099 at up to six
 *  seconds of device enumeration, so the budget is over twice the worst known
 *  real answer while staying well inside the Settings window's outer bound —
 *  the inner budget fires first and names the endpoint that hung. */
export const REQUEST_TIMEOUT_MS = 15_000;

/** The budget for the three calls that do real work rather than answer a question.
 *
 *  Transcription and model loading are legitimately slow: `POST /stt/local/load`
 *  says "may take minutes on first run (model download)", and the dictation path
 *  waits up to 300 s for local readiness before it starts transcribing at all.
 *  Ten minutes doubles that wait, and it is far outside any real transcription —
 *  uploads are capped at 25 MB, which is about thirteen minutes of 16 kHz mono
 *  audio, and the local path's own acceptance criterion is 150 s of audio in
 *  under 10 s. Unbounded was the rejected alternative: "no budget" is the defect
 *  this constant exists to remove, and it wedges the intent queue just as the
 *  short calls did, because `stopAndProcess` awaits the transcription. */
export const LONG_REQUEST_TIMEOUT_MS = 600_000;

let cachedToken: string | null = null;
let tokenPromise: Promise<string | null> | null = null;
/** The outstanding `get_backend_token` IPC call, with the time it started. When
 *  the transport hangs, `invoke()` never settles and has no reject channel
 *  (ADR 028), so the losing side of the timeout race is left pending forever. A
 *  failed token fetch is deliberately not cached — the next request must retry —
 *  which means every 5 s `/health` poll would otherwise start another one and
 *  strand it. Reusing the unsettled call keeps the retry guarantee (a
 *  *rejection* settles it, so the next round starts fresh) while cutting the
 *  strays to one per `TOKEN_CALL_REUSE_MS`. */
let pendingTokenCall: { call: Promise<string>; startedAt: number } | null = null;
let bridgeDiagnosis: BridgeDiagnosis = { kind: "ok" };
let authFailureSeen = false;

const TOKEN_EXEMPT_PATHS = new Set(["/health"]);

export function lastBridgeDiagnosis(): BridgeDiagnosis {
  return bridgeDiagnosis;
}

/** The outcome of the most recent token-gated request: `true` once one came
 *  back `401`, `false` again once one succeeds. Deliberately keyed on an
 *  observed rejection rather than on "are we inside Tauri": a backend launched
 *  without a token never returns 401, so the plain-browser dev flow is never
 *  mislabelled as broken. */
export function sawAuthFailure(): boolean {
  return authFailureSeen;
}

/** The single writer of `authFailureSeen`. Every authenticated call site
 *  reports through it — a flag that also clears has to be driven from all of
 *  them, or the badge starts disagreeing with itself. */
function recordAuthOutcome(path: string, resp: { ok: boolean; status: number }): void {
  if (TOKEN_EXEMPT_PATHS.has(path.split("?")[0])) {
    return;
  }
  if (resp.status === 401) {
    authFailureSeen = true;
  } else if (resp.ok) {
    authFailureSeen = false;
  }
}

function sharedTokenCall(start: () => Promise<string>): Promise<string> {
  const now = Date.now();
  if (pendingTokenCall !== null && now - pendingTokenCall.startedAt >= TOKEN_CALL_REUSE_MS) {
    pendingTokenCall = null;
  }
  if (pendingTokenCall === null) {
    const entry = { call: start(), startedAt: now };
    pendingTokenCall = entry;
    const release = () => {
      if (pendingTokenCall === entry) {
        pendingTokenCall = null;
      }
    };
    entry.call.then(release, release);
  }
  return pendingTokenCall.call;
}

function getToken(): Promise<string | null> {
  if (cachedToken !== null) {
    return Promise.resolve(cachedToken);
  }
  return (tokenPromise ??= (async () => {
    try {
      if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
        bridgeDiagnosis = { kind: "bridge-missing" };
        return null;
      }
      const { invoke } = await import("@tauri-apps/api/core");
      let timer: ReturnType<typeof setTimeout> | undefined;
      const expiry = new Promise<typeof TOKEN_TIMED_OUT>((resolve) => {
        timer = setTimeout(() => resolve(TOKEN_TIMED_OUT), TOKEN_TIMEOUT_MS);
      });
      let token: string | typeof TOKEN_TIMED_OUT;
      try {
        token = await Promise.race([
          sharedTokenCall(() => invoke<string>("get_backend_token")),
          expiry,
        ]);
      } finally {
        clearTimeout(timer);
      }
      if (token === TOKEN_TIMED_OUT) {
        bridgeDiagnosis = { kind: "invoke-timeout" };
        console.warn(`getToken: get_backend_token did not settle in ${TOKEN_TIMEOUT_MS} ms`);
        return null;
      }
      bridgeDiagnosis = { kind: "ok" };
      cachedToken = token;
      return cachedToken;
    } catch (err) {
      bridgeDiagnosis = {
        kind: "invoke-failed",
        detail: err instanceof Error ? err.message : String(err),
      };
      console.warn("getToken: failed to fetch backend token, will retry", err);
      return null;
    } finally {
      tokenPromise = null;
    }
  })());
}

/** Thrown on any non-401 failure, carrying the status so a caller can branch
 *  on it. The meeting-recording flow needs `403` specifically: it means the
 *  consent disclosure has not been acknowledged, which is a UI step rather
 *  than an error to report. */
export class ApiRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

async function responseError(resp: Response): Promise<Error> {
  const err = await resp.json().catch(() => ({ detail: resp.statusText }));
  const detail = err.detail || `HTTP ${resp.status}`;
  if (resp.status === 401) {
    return new ApiAuthError(detail, bridgeDiagnosis);
  }
  return new ApiRequestError(detail, resp.status);
}

/** One whole HTTP exchange under a budget, abandoning it rather than waiting
 *  forever.
 *
 *  The budget covers the body, not the headers. `fetch` settles the moment the
 *  response headers arrive, so a backend that writes headers and then stops
 *  sending leaves the caller hanging in `resp.json()` — the original defect one
 *  layer down. Disarming the controller only after the body has been consumed
 *  is what closes that: an abort during a body read rejects the `json()` promise
 *  with the same `AbortError`, which the `catch` below discriminates exactly as
 *  it does an abort during the headers.
 *
 *  The abort is also what releases the socket; the rejection it produces is
 *  translated into a `TimedOutError` so the caller gets a branchable type and a
 *  sentence naming the endpoint and the budget instead of a bare `AbortError`.
 *  The query string is dropped from that message: `/history/search?q=` carries
 *  whatever the user typed, and an error message is rendered in more places than
 *  it is read. */
async function fetchJsonWithin<T>(path: string, opts: RequestInit, budgetMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), budgetMs);
  try {
    const resp = await fetch(`${BACKEND_BASE_URL}${path}`, { ...opts, signal: controller.signal });
    recordAuthOutcome(path, resp);
    if (!resp.ok) {
      throw await responseError(resp);
    }
    return (await resp.json()) as T;
  } catch (e) {
    if (controller.signal.aborted) {
      throw new TimedOutError(budgetMs, path.split("?")[0]);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  budgetMs: number = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const token = await getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers["X-JustSay-Token"] = token;
  }
  const opts: RequestInit = { method, headers };
  if (body) {
    opts.body = JSON.stringify(body);
  }
  return fetchJsonWithin<T>(path, opts, budgetMs);
}


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

/** The meeting endpoints' own response shape. Deliberately separate from
 *  `RecordingStatus`: the dictation contract must not move, and a meeting
 *  recording has to report which output it captures (`system_endpoint`) and
 *  whether sound is arriving from it (`system_level_db`). */
export interface MeetingStatus {
  is_recording: boolean;
  duration_seconds: number;
  level_db: number;
  system_endpoint: string | null;
  system_level_db: number;
}

export interface MeetingStopResponse {
  filename: string;
  duration_seconds: number;
  truncated: boolean;
}

export interface DictateResponse {
  text: string;
  duration_ms: number;
  copied_to_clipboard: boolean;
  model_name?: string;
  fallback_reason?: string | null;
  /** Set (currently only to "silence") when the backend's silence guard
   *  short-circuited before any provider ran — no STT call, no clipboard
   *  write, no History row. Not an error: computeDoneStatus renders it via
   *  the normal "done" state. */
  discarded_reason?: string | null;
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
  /** Audio duration (seconds) at or below which the pipeline picks Groq Whisper
   *  in CLOUD mode; longer clips route to Gemini. */
  cloud_routing_threshold: number;
  /** Custom vocabulary / glossary. Plumbed into every STT provider — see the
   *  Python `STTSettings.initial_prompt` docstring for per-provider semantics.
   *  Backend enforces a 500-char ceiling. */
  initial_prompt: string;
  /** Cloud API keys. Always returned as `MASKED_API_KEY` (set) or `""` (not set)
   *  by GET/PUT. Send the real key to set it; sending `MASKED_API_KEY` back is a
   *  no-op (backend ignores it). */
  gemini_api_key: string;
  groq_api_key: string;
  /** Whether the user has acknowledged the meeting-recording disclosure. The
   *  backend answers `403` to `POST /audio/meeting/start` until it is true. */
  meeting_consent_acknowledged: boolean;
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
  /** "apple" on macOS arm64, else the detected vendor ("nvidia"/"amd"/"intel"/"none").
   *  Populated even when gpu_available is false (e.g. an explicit CPU device
   *  override) — AMD/Intel Windows and Apple Silicon both run the same
   *  whisper.cpp server provider, accelerated by Vulkan and Metal
   *  respectively; NVIDIA and CPU-only hosts use faster-whisper. */
  gpu_vendor: string;
  device: string;
  compute_type: string;
  last_error: string | null;
}

export interface GpuInfo {
  name: string;
  vendor: string;
  vram_total_mb: number;
  /** Only populated via the torch.cuda detection source — null for the
   *  Windows-registry AMD/Intel source, which has no live-usage reading. */
  vram_used_mb: number | null;
  vram_free_mb: number | null;
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
  temp_size_bytes: number;
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


export const api = {
  health: () => request<HealthResponse>("GET", "/health"),

  audioStart: () => request<RecordingStatus>("POST", "/audio/start"),

  audioStop: () => request<{ filename: string; duration_seconds: number }>("POST", "/audio/stop"),

  audioStatus: () => request<RecordingStatus>("GET", "/audio/status"),

  startMeetingRecording: () => request<MeetingStatus>("POST", "/audio/meeting/start"),

  stopMeetingRecording: () => request<MeetingStopResponse>("POST", "/audio/meeting/stop"),

  getMeetingStatus: () => request<MeetingStatus>("GET", "/audio/meeting/status"),

  dictate: (language = "uk") =>
    request<DictateResponse>(
      "POST",
      `/pipeline/dictate?language=${language}`,
      undefined,
      LONG_REQUEST_TIMEOUT_MS,
    ),

  /** Upload an audio file to the pipeline. Accepts an ArrayBuffer of file bytes.
   *  `language` defaults to `"auto"` — every STT provider maps that sentinel
   *  onto its own native auto-detect mechanism (see `STTProvider.transcribe`'s
   *  docstring in the backend for the per-provider translation). */
  processFile: async (
    fileBytes: ArrayBuffer,
    filename: string,
    language = "auto",
  ): Promise<DictateResponse> => {
    const form = new FormData();
    const blob = new Blob([fileBytes], { type: "application/octet-stream" });
    form.append("file", blob, filename);
    const path = `/pipeline/process-file?language=${language}`;
    const token = await getToken();
    const headers: Record<string, string> = {};
    if (token) {
      headers["X-JustSay-Token"] = token;
    }
    return fetchJsonWithin<DictateResponse>(
      path,
      { method: "POST", body: form, headers },
      LONG_REQUEST_TIMEOUT_MS,
    );
  },

  setSttMode: (mode: "cloud" | "local") =>
    request("PUT", "/stt/mode", { mode }),

  resources: () => request<ResourceInfo>("GET", "/resources"),

  sttLocalStatus: () => request<LocalSttStatus>("GET", "/stt/local/status"),
  sttLocalLoad: () =>
    request<{ loaded: boolean; model?: string }>(
      "POST",
      "/stt/local/load",
      undefined,
      LONG_REQUEST_TIMEOUT_MS,
    ),
  sttLocalUnload: () => request<{ unloaded: boolean }>("POST", "/stt/local/unload"),
  /** Retry affordance for the Local STT status indicator's error state —
   *  fire-and-forget on the backend, returns before the model finishes loading. */
  sttLocalPrewarm: () => request<{ started: boolean }>("POST", "/stt/local/prewarm"),

  getSettings: () => request<UserSettings>("GET", "/settings"),

  updateSettings: (updates: Partial<UserSettings>) =>
    request<SettingsUpdateResponse>("PUT", "/settings", updates),

  getStorageInfo: () => request<StorageInfo>("GET", "/settings/storage"),

  cleanupTemp: () => request<CleanupResult>("POST", "/settings/cleanup"),

  cloudKeyStatus: () => request<CloudKeyStatus>("GET", "/settings/cloud-status"),

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

  wordsTop: (lang: "all" | "uk" | "en" = "all", limit = 50) =>
    request<TopWordsResponse>("GET", `/words/top?lang=${lang}&limit=${limit}`),
};


export interface LevelStreamEvent {
  level_db: number;
  is_recording: boolean;
}

const LEVEL_STREAM_PATH = "/audio/level-stream";

/** The level meter's stream, opened under a budget that covers the handshake
 *  only.
 *
 *  The stream itself must stay unbounded — it is long-lived by design and a
 *  budget on it would cut the meter off mid-recording. The handshake is a
 *  different thing: it is opened right after `POST /audio/start`, so it can land
 *  on a backend that has stopped answering, and without a bound the meter sits
 *  flat forever with nothing on screen saying why. The timer is cleared once the
 *  reader has been obtained, which is the moment the stream begins.
 *
 *  `timedOut` is what lets the terminal `catch` tell our own abort from the
 *  caller's: `stopLevelStream()` aborts the same controller on every normal stop
 *  and must stay silent, while the handshake expiring must reach `onError`. */
export function levelStream(
  onLevel: (data: LevelStreamEvent) => void,
  onDone: () => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();
  let timedOut = false;
  const handshakeTimer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  getToken()
    .then((token) => {
      const headers: Record<string, string> = {};
      if (token) {
        headers["X-JustSay-Token"] = token;
      }
      return fetch(`${BACKEND_BASE_URL}${LEVEL_STREAM_PATH}`, {
        method: "GET",
        signal: controller.signal,
        headers,
      });
    })
    .then(async (resp) => {
      recordAuthOutcome(LEVEL_STREAM_PATH, resp);
      if (!resp.ok || !resp.body) {
        clearTimeout(handshakeTimer);
        onError(`HTTP ${resp.status}`);
        return;
      }
      const reader = resp.body.getReader();
      clearTimeout(handshakeTimer);
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
              const data: LevelStreamEvent = JSON.parse(line.slice(6));
              if (currentEvent === "done") {
                onDone();
              } else {
                onLevel(data);
              }
            } catch {
            }
          }
        }
      }
    })
    .catch((err) => {
      clearTimeout(handshakeTimer);
      if (timedOut) {
        onError(new TimedOutError(REQUEST_TIMEOUT_MS, LEVEL_STREAM_PATH).message);
        return;
      }
      if (err.name !== "AbortError") {
        onError(String(err));
      }
    });

  return controller;
}
