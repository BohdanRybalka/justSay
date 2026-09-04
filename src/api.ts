/**
 * HTTP client for JustSay Python backend.
 */

import { BACKEND_BASE_URL } from "./contracts";
import { TimedOutError } from "./timeout";

/** Why the per-launch token could not be obtained, retained so the UI can name
 *  the failing layer instead of presenting as a dead window (ADR 028).
 *  `bridge-missing` = Tauri's injected bridge scripts never ran (a synchronous
 *  throw); `bridge-timeout` = the bridge module's dynamic import never resolved,
 *  so no `invoke` was ever reached; `bridge-failed` = that same import *rejected*,
 *  which is a different fact and used to be reported as `invoke-failed` even
 *  though no `invoke` had been attempted; `invoke-timeout` = the bridge loaded
 *  but the IPC transport never answered; `invoke-failed` = the command itself
 *  rejected. The five are distinguishable on purpose: they point at different
 *  layers, and on macOS there is nothing else to attach to. */
export type BridgeDiagnosis =
  | { kind: "ok" }
  | { kind: "bridge-missing" }
  | { kind: "bridge-timeout" }
  | { kind: "bridge-failed"; detail: string }
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

/** The budget for each of the two steps that stand between a caller and the
 *  per-launch token: loading the Tauri bridge module, and the `invoke()` behind
 *  it. Both are bounded because either can go absent rather than slow — a
 *  dynamic import has no timeout and `invoke()` has no reject channel at all
 *  (ADR 028) — and an unbounded one leaves the memoised `tokenPromise` pending
 *  for the life of the window, which every later caller then joins. */
const TOKEN_TIMEOUT_MS = 3000;
const TOKEN_TIMED_OUT = Symbol("token-timed-out");
/** How long a single unanswered `get_backend_token` call may keep being reused
 *  before a fresh one is started anyway. Reuse alone would turn a *recoverable*
 *  wedge — one dropped response on an otherwise live transport — into a
 *  permanent one, because a call that never settles would be raced forever and
 *  no new `invoke()` would ever be attempted. */
export const TOKEN_CALL_REUSE_MS = 60_000;

/** How long a read may go unanswered before it is abandoned.
 *
 *  `fetch` has no timeout of its own, so a request the backend accepts and then
 *  abandons never settles — and neither does anything sequenced behind it. That
 *  is not a slow window, it is a dead one: the Settings window never reaches
 *  the failure screen its own 40 s bound was added to guarantee.
 *
 *  15 s rather than a rounder number: every read this app makes answers from
 *  memory or from one SQLite query, so the budget is an order of magnitude
 *  above the work, and it has to stay clear of `getToken()` as well — the timer
 *  is armed before the token is asked for, and `getToken()` bounds two steps in
 *  sequence, so a budget at or below `TOKEN_TIMEOUT_MS * 2` would expire with
 *  no request issued and blame the backend for a bridge fault. */
export const REQUEST_TIMEOUT_MS = 15_000;

/** The budget for a call that opens a capture device before it answers.
 *
 *  Sized to the work rather than to the transport, which is the correction ADR
 *  049's second amendment demanded. Spec 099 measured up to six seconds of
 *  device enumeration inside a single `POST /audio/meeting/start`
 *  (`specs/099-meeting-start-freezes-the-app/plan.md`), so a read's 15 s is only
 *  2.5x the worst known *healthy* answer and a slow machine reaches it with
 *  nothing wrong. A minute keeps the same order-of-magnitude margin over the
 *  measurement that `REQUEST_TIMEOUT_MS` keeps over a read. */
export const DEVICE_TIMEOUT_MS = 60_000;

/** The budget for the calls that transcribe or write a whole recording.
 *
 *  Transcription is legitimately slow: the dictation path waits up to 300 s for
 *  local readiness before it starts transcribing at all, and ten minutes
 *  doubles that. It is far outside any real answer — an upload is capped at
 *  25 MB, about thirteen minutes of 16 kHz mono audio, and the local path's own
 *  acceptance criterion is 150 s of audio in under 10 s — and it also covers
 *  `POST /audio/meeting/stop`, which writes the whole WAV before answering
 *  (`_assemble_and_write` puts a 45-minute call at ~86 MB, and
 *  `meeting_max_raw_bytes` allows roughly eight times that). */
export const LONG_REQUEST_TIMEOUT_MS = 600_000;

/** Which row of ADR 049's third amendment a call is on.
 *
 *  The rule there is a comparison rather than a cost: a budget belongs where
 *  abandoning the request costs the user *less* than waiting for it forever
 *  does. Every entry in `api` states its own row, and there is no default, so a
 *  new endpoint has to be placed rather than inherit a number nobody chose. */
type Budget = { readonly ms: number } | { readonly ms: null };

/** A read the UI can simply ask for again. Abandoning costs one more press;
 *  waiting forever costs a screen that never resolves. */
const REREADABLE: Budget = { ms: REQUEST_TIMEOUT_MS };

/** A call that opens a capture device before it answers. Abandoning it is a
 *  strict subset of waiting on it: the device may be open either way, and only
 *  waiting adds a dead intent queue, or a toggle that swallows every press
 *  including the tray's, on top of that. */
const DEVICE_TRANSITION: Budget = { ms: DEVICE_TIMEOUT_MS };

/** The same comparison over work measured in minutes rather than seconds —
 *  transcribing a dictation, writing a meeting's WAV to disk. */
const SLOW_TRANSITION: Budget = { ms: LONG_REQUEST_TIMEOUT_MS };

/** A call where the comparison runs the other way. Abandoning reports a failure
 *  for work that ran to completion anyway and invites a second destructive
 *  call, or throws away a backend degradation path that had already half
 *  answered; waiting costs a spinner. Giving up here is only honest once the
 *  client can find out what the backend did, and that reconciliation needs the
 *  client-minted session id spec 119 adds. Until then no budget, rather than a
 *  wrong one. */
const UNRECONCILED: Budget = { ms: null };

let cachedToken: string | null = null;
let tokenPromise: Promise<string | null> | null = null;
/** The outstanding `get_backend_token` IPC call and the outstanding bridge
 *  import, each with the time it started. Both steps can go absent rather than
 *  slow — `invoke()` has no reject channel at all (ADR 028) and a dynamic import
 *  of a module that never arrives stays pending for the life of the window — so
 *  the losing side of either timeout race is left attached forever. A failed
 *  token fetch is deliberately not cached, because the next request must retry,
 *  which means every 5 s `/health` poll would otherwise start another one and
 *  strand it. Reusing the unsettled call keeps the retry guarantee (a
 *  *rejection* settles it, so the next round starts fresh) while cutting the
 *  strays to one per `TOKEN_CALL_REUSE_MS`.
 *
 *  The import needs it for the same reason the invoke does and did not have it:
 *  `import()` hands every caller the *same* pending module promise, so each new
 *  `getToken()` attached one more never-releasable reaction pair to it and
 *  logged one more warning about a fault already reported. */
type CallSlot<T> = { pending: { call: Promise<T>; startedAt: number } | null };
const tokenCallSlot: CallSlot<string> = { pending: null };
type BridgeModule = typeof import("@tauri-apps/api/core");
const bridgeImportSlot: CallSlot<BridgeModule> = { pending: null };
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

/** `fresh` says whether this caller is the one that started the underlying
 *  call, so a diagnosis that is logged once per fault can tell itself apart
 *  from the callers merely joining it. */
function shareWhilePending<T>(
  slot: CallSlot<T>,
  start: () => Promise<T>,
  reuseMs: number,
): { call: Promise<T>; fresh: boolean } {
  const now = Date.now();
  if (slot.pending !== null && now - slot.pending.startedAt >= reuseMs) {
    slot.pending = null;
  }
  if (slot.pending !== null) {
    return { call: slot.pending.call, fresh: false };
  }
  const entry = { call: start(), startedAt: now };
  slot.pending = entry;
  const release = () => {
    if (slot.pending === entry) {
      slot.pending = null;
    }
  };
  entry.call.then(release, release);
  return { call: entry.call, fresh: true };
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
      let importTimer: ReturnType<typeof setTimeout> | undefined;
      const importExpiry = new Promise<typeof TOKEN_TIMED_OUT>((resolve) => {
        importTimer = setTimeout(() => resolve(TOKEN_TIMED_OUT), TOKEN_TIMEOUT_MS);
      });
      const bridgeImport = shareWhilePending(
        bridgeImportSlot,
        () => import("@tauri-apps/api/core"),
        TOKEN_CALL_REUSE_MS,
      );
      let bridge: BridgeModule | typeof TOKEN_TIMED_OUT;
      try {
        bridge = await Promise.race([bridgeImport.call, importExpiry]);
      } catch (e) {
        bridgeDiagnosis = {
          kind: "bridge-failed",
          detail: e instanceof Error ? e.message : String(e),
        };
        console.warn("getToken: the Tauri bridge module failed to load", e);
        return null;
      } finally {
        clearTimeout(importTimer);
      }
      if (bridge === TOKEN_TIMED_OUT) {
        bridgeDiagnosis = { kind: "bridge-timeout" };
        if (bridgeImport.fresh) {
          console.warn(`getToken: the Tauri bridge module did not load in ${TOKEN_TIMEOUT_MS} ms`);
        }
        return null;
      }
      const { invoke } = bridge;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const expiry = new Promise<typeof TOKEN_TIMED_OUT>((resolve) => {
        timer = setTimeout(() => resolve(TOKEN_TIMED_OUT), TOKEN_TIMEOUT_MS);
      });
      let token: string | typeof TOKEN_TIMED_OUT;
      try {
        token = await Promise.race([
          shareWhilePending(
            tokenCallSlot,
            () => invoke<string>("get_backend_token"),
            TOKEN_CALL_REUSE_MS,
          ).call,
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

/** The error a non-2xx response describes — or the abort, rethrown.
 *
 *  Reading the error body can itself run out of the budget, and a blanket
 *  `.catch` here turns that abort into a fully formed `ApiRequestError`, which
 *  `fetchJsonWithin`'s catch then has no way to recognise as a timeout. The
 *  caller ends up branching on a status the backend never finished sending.
 *
 *  The body is read as nullable because `resp.json()` *succeeding* does not mean
 *  it produced an object: a response whose body is the four bytes `null` is
 *  valid JSON and resolves to `null`, and reading `.detail` off it threw a
 *  `TypeError` that left this function through neither of its two paths — the
 *  caller saw a type error where it was branching on a status. */
async function responseError(resp: Response): Promise<Error> {
  let err: { detail?: string } | null;
  try {
    err = await resp.json();
  } catch (e) {
    if (isAbortError(e)) throw e;
    err = { detail: resp.statusText };
  }
  const detail = err?.detail || `HTTP ${resp.status}`;
  if (resp.status === 401) {
    return new ApiAuthError(detail, bridgeDiagnosis);
  }
  return new ApiRequestError(detail, resp.status);
}

/** An abort is identified by its `name`, not by its class.
 *
 *  `fetch` rejects an aborted request with a `DOMException`, and `DOMException`
 *  does not extend `Error` in every environment this code runs in — it does not
 *  under jsdom, which is what the suite uses. Discriminating on the name is
 *  what makes the check hold in the browser and in the tests both, and the
 *  check has to discriminate: `signal.aborted` alone relabels every error
 *  raised after the budget expired, including a `403` whose body merely stopped
 *  part-way, and a `403` reported as a timeout is a refusal the caller can no
 *  longer act on — the consent dialog it should open never opens. */
function isAbortError(e: unknown): boolean {
  return typeof e === "object" && e !== null && (e as { name?: unknown }).name === "AbortError";
}

/** Await `work`, but give up the moment `signal` aborts.
 *
 *  `getToken()` takes no signal, so arming a timer in front of it does nothing
 *  on its own — nothing is listening — and the wait itself has to end when the
 *  budget does. The abandoned work is left running; it settles or it does not,
 *  and either way no caller is still attached to it once it has.
 *
 *  *Once it has* is the load-bearing half. This function detaches by hanging
 *  its own cleanup off `work`, so a `work` that never settles keeps every
 *  abandoned caller's `resolve`/`reject` pair attached to it, and nothing this
 *  wrapper can do reaches them while it is still pending. */
function until<T>(work: Promise<T>, signal: AbortSignal, giveUp: () => Error): Promise<T> {
  if (signal.aborted) return Promise.reject(giveUp());
  return new Promise<T>((resolve, reject) => {
    const stop = () => reject(giveUp());
    signal.addEventListener("abort", stop, { once: true });
    work.then(resolve, reject).finally(() => signal.removeEventListener("abort", stop));
  });
}

/** The exchange itself, with no opinion about how long it may take. */
async function fetchJson<T>(path: string, opts: RequestInit): Promise<T> {
  const resp = await fetch(`${BACKEND_BASE_URL}${path}`, opts);
  recordAuthOutcome(path, resp);
  if (!resp.ok) {
    throw await responseError(resp);
  }
  return (await resp.json()) as T;
}

/** One whole HTTP exchange under a budget, abandoning it rather than waiting
 *  forever.
 *
 *  The budget covers the token wait and the body, not only the headers.
 *  `fetch` settles the moment the response headers arrive, so a backend that
 *  writes headers and then stops
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
async function fetchJsonWithin<T>(
  path: string,
  buildRequest: () => Promise<RequestInit>,
  budgetMs: number,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), budgetMs);
  const timedOut = () => new TimedOutError(budgetMs, path.split("?")[0]);
  try {
    const opts = await until(buildRequest(), controller.signal, timedOut);
    return await fetchJson<T>(path, { ...opts, signal: controller.signal });
  } catch (e) {
    if (e instanceof TimedOutError) {
      throw e;
    }
    if (isAbortError(e) && controller.signal.aborted) {
      throw timedOut();
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/** The same exchange with no budget and therefore no `AbortController`: an
 *  `UNRECONCILED` call is waited out, because abandoning it would leave the UI
 *  asserting an outcome nobody established. */
async function fetchJsonUntilAnswered<T>(
  path: string,
  buildRequest: () => Promise<RequestInit>,
): Promise<T> {
  return fetchJson<T>(path, await buildRequest());
}

/** The one place a `Budget` decides which of the two mechanisms runs, so a call
 *  that builds its own request — `processFile` and its `FormData` — is placed
 *  by the same object every other endpoint is placed by. */
function send<T>(
  path: string,
  buildRequest: () => Promise<RequestInit>,
  budget: Budget,
): Promise<T> {
  return budget.ms === null
    ? fetchJsonUntilAnswered<T>(path, buildRequest)
    : fetchJsonWithin<T>(path, buildRequest, budget.ms);
}

/** `budget` is an object rather than a fourth positional argument because the
 *  third is `body?: unknown`: a bare number in that slot type-checks, ships the
 *  budget as the request body, and silently keeps the default budget. It has no
 *  default at all, so every endpoint below names the class it is in. */
function request<T>(
  method: string,
  path: string,
  body: unknown,
  budget: Budget,
): Promise<T> {
  const buildRequest = async () => {
    const token = await getToken();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["X-JustSay-Token"] = token;
    }
    const opts: RequestInit = { method, headers };
    if (body) {
      opts.body = JSON.stringify(body);
    }
    return opts;
  };
  return send<T>(path, buildRequest, budget);
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


/**
 * Every endpoint, each stating which row of ADR 049's third amendment it is on.
 *
 * The question each answers is the comparison, not the cost: is abandoning this
 * request worse for the user than waiting for it forever? For a `REREADABLE`
 * read the answer is trivially no. For a `DEVICE_TRANSITION` or a
 * `SLOW_TRANSITION` it is no because abandoning is a strict subset of waiting —
 * the device is in the same state either way, and only waiting adds a dead
 * intent queue or a dead toggle on top of it. For an `UNRECONCILED` call the
 * answer is yes: abandoning reports a failure for work that finished, or throws
 * away a half-answered degradation path, while waiting costs a spinner.
 *
 * What a budget does not buy is the right to guess. The callers of the two
 * transition classes report the abandonment as its own outcome and never as a
 * failure that was observed.
 */
export const api = {
  health: () => request<HealthResponse>("GET", "/health", undefined, REREADABLE),

  audioStatus: () => request<RecordingStatus>("GET", "/audio/status", undefined, REREADABLE),

  getMeetingStatus: () =>
    request<MeetingStatus>("GET", "/audio/meeting/status", undefined, REREADABLE),

  resources: () => request<ResourceInfo>("GET", "/resources", undefined, REREADABLE),

  sttLocalStatus: () => request<LocalSttStatus>("GET", "/stt/local/status", undefined, REREADABLE),

  getSettings: () => request<UserSettings>("GET", "/settings", undefined, REREADABLE),

  getStorageInfo: () => request<StorageInfo>("GET", "/settings/storage", undefined, REREADABLE),

  cloudKeyStatus: () =>
    request<CloudKeyStatus>("GET", "/settings/cloud-status", undefined, REREADABLE),

  getHistory: (limit = 50, offset = 0) =>
    request<HistoryListResponse>(
      "GET",
      `/history?limit=${limit}&offset=${offset}`,
      undefined,
      REREADABLE,
    ),

  historyStats: () => request<HistoryStats>("GET", "/history/stats", undefined, REREADABLE),

  wordsTop: (lang: "all" | "uk" | "en" = "all", limit = 50) =>
    request<TopWordsResponse>("GET", `/words/top?lang=${lang}&limit=${limit}`, undefined, REREADABLE),

  /** A read, and still `UNRECONCILED`: `_semantic_lane` falls back to full-text
   *  search when the embedding provider is slow (ADR 010), and a slow provider
   *  does not raise, so a client-side budget fires first and throws away the
   *  local half that had already answered. A slow search would become no
   *  search — the degradation path the backend was built with, preempted. */
  searchHistory: (q: string, limit = 30) =>
    request<HistoryListResponse>(
      "GET",
      `/history/search?q=${encodeURIComponent(q)}&limit=${limit}`,
      undefined,
      UNRECONCILED,
    ),

  /** `POST /audio/start` calls `await recorder.start()` before it answers, so
   *  the microphone may be open whichever way this ends. Waiting adds the widget
   *  stuck on "Recording" and an intent queue parked inside the start — the
   *  wedge this spec exists to remove — so the budget is the smaller cost, and
   *  the caller says the request was abandoned rather than that the start
   *  failed. */
  audioStart: () => request<RecordingStatus>("POST", "/audio/start", undefined, DEVICE_TRANSITION),

  audioStop: () =>
    request<{ filename: string; duration_seconds: number }>(
      "POST",
      "/audio/stop",
      undefined,
      UNRECONCILED,
    ),

  /** Opens both devices before answering, which is why it is on the device
   *  budget rather than a read's — spec 099 measured up to six seconds of
   *  enumeration for this one call. Waiting forever is what left the toggle
   *  swallowing every press, the tray's included, for the life of the window. */
  startMeetingRecording: () =>
    request<MeetingStatus>("POST", "/audio/meeting/start", undefined, DEVICE_TRANSITION),

  /** Writes the whole WAV before it answers — `_assemble_and_write` puts a
   *  45-minute call at "~86 MB to disk" and `meeting_max_raw_bytes` allows
   *  roughly eight times that — so it takes the long budget rather than the
   *  device one. The recording survives the abort; what an abandoned stop costs
   *  is the answer, and the indicator stays up on it. */
  stopMeetingRecording: () =>
    request<MeetingStopResponse>("POST", "/audio/meeting/stop", undefined, SLOW_TRANSITION),

  /** Stops the recorder as the handler's first act and can write the clipboard
   *  as its last — but a backend that never ran the handler stopped nothing, so
   *  the microphone may be open either way and only waiting adds a widget stuck
   *  on "Processing" to it. The long budget, because transcription is the work
   *  being waited on rather than a device open. */
  dictate: (language = "uk") =>
    request<DictateResponse>(
      "POST",
      `/pipeline/dictate?language=${language}`,
      undefined,
      SLOW_TRANSITION,
    ),

  /** Upload an audio file to the pipeline. Accepts an ArrayBuffer of file bytes.
   *  `language` defaults to `"auto"` — every STT provider maps that sentinel
   *  onto its own native auto-detect mechanism (see `STTProvider.transcribe`'s
   *  docstring in the backend for the per-provider translation).
   *
   *  `SLOW_TRANSITION` for the same reason as `dictate`, and the same number
   *  covers its own worst case: the upload is capped at 25 MB, about thirteen
   *  minutes of 16 kHz mono audio. It opens no device, so waiting forever costs
   *  only the Transcribe tab — but that tab has no other way out, which is the
   *  comparison this class is on. */
  processFile: (
    fileBytes: ArrayBuffer,
    filename: string,
    language = "auto",
  ): Promise<DictateResponse> => {
    const form = new FormData();
    const blob = new Blob([fileBytes], { type: "application/octet-stream" });
    form.append("file", blob, filename);
    return send<DictateResponse>(
      `/pipeline/process-file?language=${language}`,
      async () => {
        const token = await getToken();
        const headers: Record<string, string> = {};
        if (token) {
          headers["X-JustSay-Token"] = token;
        }
        return { method: "POST", body: form, headers };
      },
      SLOW_TRANSITION,
    );
  },

  setSttMode: (mode: "cloud" | "local") => request("PUT", "/stt/mode", { mode }, UNRECONCILED),

  /** "May take minutes on first run (model download)", and it leaves a loaded
   *  provider behind whether or not this window waited for the answer. */
  sttLocalLoad: () =>
    request<{ loaded: boolean; model?: string }>(
      "POST",
      "/stt/local/load",
      undefined,
      UNRECONCILED,
    ),

  sttLocalUnload: () =>
    request<{ unloaded: boolean }>("POST", "/stt/local/unload", undefined, UNRECONCILED),

  /** Retry affordance for the Local STT status indicator's error state —
   *  fire-and-forget on the backend, returns before the model finishes loading.
   *  There is nothing to reconcile, so it is a read for this purpose: abandoning
   *  it leaves the load running exactly as answering it would, while waiting
   *  forever hangs the one button that exists to escape a stuck state. */
  sttLocalPrewarm: () =>
    request<{ started: boolean }>("POST", "/stt/local/prewarm", undefined, REREADABLE),

  updateSettings: (updates: Partial<UserSettings>) =>
    request<SettingsUpdateResponse>("PUT", "/settings", updates, UNRECONCILED),

  /** Unlinks every scratch file inline and is bounded by nothing on the
   *  backend, so a budget would report a failure for a deletion that completes
   *  — and the natural retry is a second destructive call issued against a
   *  backend still executing the first. */
  cleanupTemp: () => request<CleanupResult>("POST", "/settings/cleanup", undefined, UNRECONCILED),

  deleteHistoryEntry: (id: string) =>
    request<{ deleted: boolean }>("DELETE", `/history/${id}`, undefined, UNRECONCILED),

  /** Deletes the transcripts and their `entry_embeddings` vector rows, with no
   *  backend bound on either — the same false report about deleted data that
   *  `cleanupTemp` would give. */
  clearHistory: () => request<{ deleted: number }>("DELETE", "/history", undefined, UNRECONCILED),
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
 *  flat forever with nothing on screen saying why.
 *
 *  The timer is cleared once the first chunk has been read, not once the reader
 *  has been obtained. `getReader()` is available the instant the response
 *  *headers* arrive, so clearing there bounds exactly what `fetch` already
 *  settles on, and a backend that writes `200 text/event-stream` and then stops
 *  sending sits in the same flat-meter silence this budget exists to break —
 *  the headers-only bound `fetchJsonWithin` refuses one layer down.
 *
 *  A handshake is `REREADABLE` in ADR 049's sense: abandoning it moves nothing,
 *  and the remedy is to open the stream again.
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

  until(getToken(), controller.signal, () =>
    timedOut
      ? new TimedOutError(REQUEST_TIMEOUT_MS, LEVEL_STREAM_PATH)
      : new DOMException("The level stream was closed by its caller.", "AbortError"),
  )
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
      const decoder = new TextDecoder();
      let buffer = "";
      let handshakeDone = false;

      while (true) {
        const { done, value } = await reader.read();
        if (!handshakeDone) {
          handshakeDone = true;
          clearTimeout(handshakeTimer);
        }
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
      if (err instanceof TimedOutError) {
        onError(err.message);
        return;
      }
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
