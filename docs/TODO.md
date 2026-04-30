# TODO

Task tracker for the JustSay project. Tasks are organized by status.
See `docs/plans/` for detailed plans per initiative.

## In Progress

- [ ] Local mode UX (Plan 004) — Guardian Toggle
  - [x] Phase 4.1: Backend setup modules (status endpoints, tests)
  - [x] Phase 4.2: Backend download/pull endpoints (SSE streaming, Ollama auto-start)
  - [x] Phase 4.3: Frontend Guardian UI (inline status, progress bar, SSE client)
  - [x] Phase 4.4: Provider cleanup, shutdown hooks, resource monitoring, sync_to_runtime
  - [x] Phase 4.5: State machine redesign — explicit Start/Stop, polling, factory cache by model key
  - [ ] Phase 4.6: Update default local models for macOS target (large-v3-turbo, qwen3:1.7b)

## Backlog

- [ ] Local model optimization for macOS Apple Silicon (whisper.cpp Metal, Ollama Metal)
  - [ ] Replace faster-whisper with whisper.cpp (Metal GPU support for Apple Silicon)
  - [ ] Set default local models: large-v3-turbo (STT) + qwen3:1.7b (LLM)
  - [ ] Benchmark on M1/M2 8GB: target 1-2.5s short audio, ≤10s long audio
  - [ ] Update beam_size=1 for faster STT in low-latency mode
- [ ] API Keys management UI — test/validate keys from Settings (Gemini + Groq STT + Groq LLM)
- [ ] macOS/iOS port planning — native CoreML/MLX engine for iOS
- [ ] Move `docs/hybrid-stt-pipeline.md` → `docs/plans/005-hybrid-stt-pipeline.md` (per CLAUDE.md convention: plans live in `docs/plans/`)


## Tech Debt

### Security
- [ ] Thread-safe provider caching — race condition при concurrent requests
- [ ] Thread-safe lazy init в `_get_model()` / `_get_client()`
- [ ] MIME type validation для /stt/transcribe — magic bytes замість розширення

### Architecture
- [ ] Migrate audio singleton to FastAPI DI + lifespan
- [x] Provider `cleanup()` method — explicit звільнення GPU пам'яті (done in Phase 4.4)
- [ ] DRY factory pattern — витягти в generic `CachedProviderFactory[T]`
- [ ] Backend watchdog — респаун Python при краші
- [ ] Drop impl для BACKEND_PROCESS — cleanup при Tauri panic
- [ ] Python version check — find_python() не перевіряє >=3.10
- [ ] Graceful backend shutdown — SIGTERM → wait → SIGKILL

### Frontend
- [ ] Reqwest Client reuse — lazy_static замість нового клієнта на кожен запит
- [x] UI для зміни хоткея — реалізовано в Settings > General
- [x] UI для вибору мови — реалізовано в Settings > General
- [ ] Notification/toast при помилках замість тільки badge
- [ ] Індикація LLM degradation (коли fallback до raw_text)
- [ ] Double-press guard — asyncio race при швидкому подвійному натисканні

### UX
- [ ] WebSocket для audio level_db streaming (замість polling)
- [ ] Logging у провайдерах — спостережуваність при помилках API
- [ ] Gemini safety filter handling — response.text ValueError

## Done

- [x] Phase 1.1: Python Backend skeleton — FastAPI, config, abstract providers, tests
- [x] Phase 1.1 fixes: QA/Architect review — CORS, version, .env.example, test isolation, domain model
- [x] Refactoring: modular Lego-block architecture (core/, stt/, llm/, audio/, pipeline/)
- [x] Phase 2: Audio Capture — MicrophoneRecorder, /audio/* endpoints, thread-safe, WAV output
- [x] Phase 2 fixes: race condition, try/finally, path leak, HTTP 409, max duration, config validation
- [x] Phase 3: STT providers — Cloud (Gemini), Local (faster-whisper), provider caching, /stt/transcribe
- [x] Phase 3 fixes: path traversal, file size limit, broad exception handling
- [x] Phase 4: LLM providers — Cloud (Groq), Local (Ollama), /llm/process, input size limits
- [x] Phase 5: Pipeline endpoints — /pipeline/dictate, /pipeline/process-file, clipboard, graceful degradation
- [x] Phase 1.2: Tauri Frontend — system tray, Ctrl+Alt+V push-to-talk, dark UI, mode switching
- [x] Phase 1.2 fixes: CSP, reqwest timeout+status check, recorder singleton accessor, dev CORS, shell:allow-open removed
- [x] Phase 1.3: Integration — Tauri spawns Python backend, health poll, graceful shutdown, Python discovery
- [x] Phase 1.3 fixes: Stdio::null (deadlock prevention), port check, Windows taskkill /T process tree, try_wait during health poll
- [x] Phase 2.1: Widget — Dynamic Island pill widget with SetWindowRgn, click toggle, push-to-talk, marquee result
- [x] Phase 2.2: Widget fixes — remove WebView2 transparency attempts, solid pill shape via Win32 region
- [x] Phase 3.1: Settings Panel — Shell, Navigation, UserSettings backend
- [x] Phase 3.2: General + Models tabs
- [x] Phase 3.3: Audio tab with mic test & dictation
- [x] Phase 3.4: Storage tab with dir picker & cleanup
- [x] Phase 3.5: Widget improvements — logo, custom shortcut, history tab, transcription style selector
- [x] Phase 3.6: Prompt engineering — code-switching, removed hardcoded fillers, STT punctuation, adaptive AI prompt, per-style temperature
- [x] Hybrid STT Pipeline (`docs/hybrid-stt-pipeline.md`, v2.0) — smart routing + LLM removal
  - [x] Task 6: Config — `groq_api_key`, `groq_whisper_model`, `cloud_routing_threshold` (+ `>0` validator), per-provider format whitelists (`.webm` excluded from Groq), `UserSettings.cloud_routing_threshold` + `sync_to_runtime`
  - [x] Task 1: `GroqWhisperSTTProvider` (`whisper-large-v3-turbo`, `response_format="text"`, 10s timeout, 429→clearer RuntimeError)
  - [x] Task 3: `CloudSTTProvider` → `GeminiSTTProvider` (alias kept), style-aware prompts (`normal` vs `ai_prompt`) via `**kwargs` on `STTProvider.transcribe()`, refusal/empty-response handling, safety-filter `ValueError` guard
  - [x] Task 2: `get_routed_provider()` in `stt/__init__.py` with dict-based thread-safe cache (`_cache_lock`), `recorder.last_duration_seconds` property + `min(elapsed, max_duration)` clamp, `detect_duration()` via `soundfile` in `pipeline/utils.py`, `/process-file` accepts `style`, webm-fallback to Gemini
  - [x] Task 4: LLM step removed from `pipeline/service.py` (raw=cleaned, standalone `/llm/process` intact), pipeline tests migrated to routing mocks
  - [x] Task 5: `pyproject.toml` — `ollama` moved from `[local]` to new `[local-llm]` extra (lazy import in `llm/local.py` already present, no source changes needed)
