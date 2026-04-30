# Plan 004: Local Mode UX — Guardian Toggle

## Goal

Make Local mode a first-class citizen with proper prerequisite checking, guided setup, download progress, and hardware visibility. When the user clicks "Local", the system verifies everything is ready — and if not, guides them through setup inline.

## Problem

Currently switching to Local mode silently fails if:
- Ollama is not installed or not running (LLM)
- Required model is not pulled (LLM: gemma3:4b ~3GB)
- Whisper model is not cached (STT: large-v3 ~3GB)
- GPU is unavailable (falls back to CPU without telling the user)

Zero feedback. Zero onboarding. The toggle returns 200 OK and the next dictation crashes.

## Design: "Guardian Toggle"

The Cloud/Local toggle becomes "smart" — before activating Local mode, it checks prerequisites and shows inline setup UI if anything is missing.

### User Flow

```
User clicks [Local] toggle
    ↓
┌─ Pre-flight check (GET /stt/local/status or /llm/local/status) ─┐
│                                                                   │
│  Case 1: Everything ready                                         │
│  → Toggle activates immediately                                   │
│  → Show: "● Ready — whisper/large-v3 · CUDA (RTX 3060)"         │
│                                                                   │
│  Case 2: Runtime missing (Ollama not running)                     │
│  → Toggle stays inactive                                          │
│  → Show: "○ Ollama not running"                                   │
│  → Buttons: [Start Ollama] [Check Again] [Install Guide ↗]       │
│                                                                   │
│  Case 3: Model not downloaded                                     │
│  → Toggle stays inactive                                          │
│  → Show: "○ Model gemma3:4b not found (3.1 GB)"                  │
│  → Button: [Download]                                             │
│  → On click: progress bar with streaming bytes                    │
│  → On complete: auto-activate toggle                              │
│                                                                   │
│  Case 4: Download in progress                                     │
│  → Show: "⏳ Downloading gemma3:4b"                               │
│  → Progress: ████████░░░░ 62% · 1.9 / 3.1 GB                    │
│  → Button: [Cancel]                                               │
└───────────────────────────────────────────────────────────────────┘
```

### UI Layout (Models Tab)

```
┌─ Speech-to-Text ───────────────────────────────────────┐
│  Mode:  [Cloud ●] [Local ○]                            │
│                                                        │
│  ┌─ Local Status ────────────────────────────────────┐ │
│  │  Model    whisper/large-v3         ● Downloaded    │ │
│  │  Size     2.9 GB                                  │ │
│  │  Device   NVIDIA GeForce RTX 3060 (CUDA)          │ │
│  │                                    [Re-check]     │ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘

┌─ Language Model ───────────────────────────────────────┐
│  Mode:  [Cloud ○] [Local ●]                            │
│                                                        │
│  ┌─ Local Status ────────────────────────────────────┐ │
│  │  Ollama   ● Running (v0.9.2)                      │ │
│  │  Model    gemma3:4b                ● Ready         │ │
│  │  Size     3.1 GB                                  │ │
│  │  VRAM     1.8 GB used                             │ │
│  │                                    [Re-check]     │ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

Download state:
```
│  ┌─ Local Status ────────────────────────────────────┐ │
│  │  Ollama   ● Running                               │ │
│  │  Model    gemma3:4b                ○ Not found     │ │
│  │                                                   │ │
│  │  ████████████░░░░░░░░  62%  ·  1.9 / 3.1 GB      │ │
│  │                                        [Cancel]   │ │
│  └───────────────────────────────────────────────────┘ │
```

## Architecture

### Key Principle: Setup ≠ Runtime

Setup modules check "can we run?" — runtime providers execute "run it". They are separate files.

```
backend/app/
├── stt/
│   ├── local.py            # existing runtime provider (unchanged)
│   ├── local_setup.py      # NEW: readiness checks, whisper download
│   └── router.py           # extend: add /stt/local/status, /stt/local/download
├── llm/
│   ├── local.py            # existing runtime provider (unchanged)
│   ├── local_setup.py      # NEW: Ollama health, model check, pull proxy
│   └── router.py           # extend: add /llm/local/status, /llm/local/pull
```

### Backend API

#### `GET /stt/local/status`

Check faster-whisper readiness.

```json
{
  "model_downloaded": true,
  "model_name": "large-v3",
  "model_size_bytes": 3087007744,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "device": "cuda",
  "compute_type": "float16"
}
```

Implementation:
- Check whisper model in huggingface_hub cache: `try_to_load_from_cache("Systran/faster-whisper-large-v3", "model.bin")`
- GPU: `torch.cuda.is_available()` + `torch.cuda.get_device_name(0)`
- Model size: scan cached directory size or use known size table

#### `POST /stt/local/download`

Download whisper model with progress via SSE (Server-Sent Events).

```
event: progress
data: {"status": "downloading", "completed": 1932735283, "total": 3087007744}

event: progress
data: {"status": "downloading", "completed": 3087007744, "total": 3087007744}

event: done
data: {"status": "done"}
```

Implementation:
- Use `huggingface_hub.snapshot_download()` with `tqdm_class` callback for progress
- Wrap in `StreamingResponse(media_type="text/event-stream")`

#### `GET /llm/local/status`

Check Ollama + model readiness.

```json
{
  "ollama_running": true,
  "ollama_version": "0.9.2",
  "model_downloaded": true,
  "model_name": "gemma3:4b",
  "model_size_bytes": 3341680640,
  "model_loaded": false,
  "vram_used_bytes": null
}
```

Implementation:
- Health: `GET http://localhost:11434/` (200 = running)
- Version: `GET http://localhost:11434/api/version`
- Models: `GET http://localhost:11434/api/tags` → check if target model is in list
- Loaded: `GET http://localhost:11434/api/ps` → check if model is in RAM, get `size_vram`

#### `POST /llm/local/pull`

Pull Ollama model with progress via SSE.

```
event: progress
data: {"status": "pulling manifest"}

event: progress
data: {"status": "pulling layers", "completed": 1483652928, "total": 3341680640}

event: done
data: {"status": "success"}
```

Implementation:
- Proxy `POST http://localhost:11434/api/pull` with `stream: true`
- Parse NDJSON response, re-emit as SSE events
- Handle cancellation via client disconnect

#### `POST /llm/local/start`

Attempt to start Ollama if not running.

```json
{
  "started": true,
  "method": "subprocess"
}
```

Implementation:
- `subprocess.Popen(["ollama", "serve"], creationflags=CREATE_NO_WINDOW)` on Windows
- Poll `GET /` until 200 or timeout (10s)
- Return `started: false` with `error` if `ollama` not in PATH

### Frontend Changes

#### `src/api.ts` — New types and methods

```typescript
export interface LocalSttStatus {
  model_downloaded: boolean;
  model_name: string;
  model_size_bytes: number | null;
  gpu_available: boolean;
  gpu_name: string | null;
  device: string;
  compute_type: string;
}

export interface LocalLlmStatus {
  ollama_running: boolean;
  ollama_version: string | null;
  model_downloaded: boolean;
  model_name: string;
  model_size_bytes: number | null;
  model_loaded: boolean;
  vram_used_bytes: number | null;
}

export const api = {
  // ... existing methods ...

  sttLocalStatus: () => request<LocalSttStatus>("GET", "/stt/local/status"),
  llmLocalStatus: () => request<LocalLlmStatus>("GET", "/llm/local/status"),
  llmLocalStart: () => request<{ started: boolean }>("POST", "/llm/local/start"),
};

// SSE helpers (not through request() — raw fetch with streaming)
export function sttLocalDownload(onProgress: (data: ProgressEvent) => void): AbortController { ... }
export function llmLocalPull(onProgress: (data: ProgressEvent) => void): AbortController { ... }
```

#### `src/settings/tabs/models.ts` — Guardian Logic

```
onToggleLocal(domain):
  1. Disable toggle button (loading state)
  2. Fetch status endpoint
  3. If all ready → activate toggle, show status panel
  4. If not ready → show inline setup panel:
     - Checklist of what's missing
     - Action buttons (Download, Start Ollama, etc.)
     - Progress bar when downloading
  5. After setup complete → auto-activate toggle
```

### SSE vs Polling Decision

**Use SSE** (Server-Sent Events) for download progress:
- FastAPI `StreamingResponse` supports it natively
- Frontend `EventSource` API works in Tauri webview (Chromium-based)
- Clean abort via `AbortController`
- No polling overhead for 3GB+ downloads

**Fallback plan**: If SSE proves unreliable in Tauri, switch to polling `GET /llm/local/pull-status` every 500ms. The backend would track pull state in memory.

### Safety Guards

1. **Block mode switch during recording**: Check `/audio/status` before allowing toggle. If `is_recording: true`, show "Stop recording first" message.

2. **Cancel download on settings close**: Frontend sends abort signal; backend cancels HTTP stream to Ollama / kills download thread.

3. **Timeout on Ollama operations**: All HTTP calls to Ollama get 10s connect timeout, 30s read timeout.

4. **No data leakage in Local mode**: All download traffic goes to:
   - `huggingface.co` for whisper model (one-time download, cached locally)
   - `registry.ollama.ai` for Ollama model (one-time download, cached locally)
   - After download: zero external traffic. This must be documented in UI.

## Implementation Phases

### Phase 4.1: Backend Setup Modules

Files to create:
- `backend/app/stt/local_setup.py` — whisper readiness check, model cache detection
- `backend/app/llm/local_setup.py` — Ollama health, model list, version check

Files to modify:
- `backend/app/stt/router.py` — add `GET /stt/local/status`
- `backend/app/llm/router.py` — add `GET /llm/local/status`

Tests:
- `backend/tests/test_stt_local_setup.py`
- `backend/tests/test_llm_local_setup.py`

### Phase 4.2: Backend Download/Pull Endpoints

Files to modify:
- `backend/app/stt/router.py` — add `POST /stt/local/download` (SSE)
- `backend/app/llm/router.py` — add `POST /llm/local/pull` (SSE), `POST /llm/local/start`
- `backend/app/llm/local_setup.py` — add pull proxy, Ollama auto-start

### Phase 4.3: Frontend Guardian UI

Files to modify:
- `src/api.ts` — add types + methods for local status/download/pull
- `src/settings/tabs/models.ts` — Guardian Toggle logic, inline status panel, progress bar
- `src/settings/settings.css` — styles for status panel, progress bar, badges

### Phase 4.4: Integration & Polish

- SSE test in Tauri webview (fallback to polling if needed)
- Cancel download on settings window close
- Mode switch guard during recording
- Error states: network down, disk full, Ollama crash mid-pull
- Timeout configuration

## Out of Scope (Future)

- Model selector dropdown (change from gemma3:4b to another model)
- Whisper model size selector with size comparison
- Auto-update models (check for newer versions)
- GPU memory estimation ("this model needs X GB, you have Y GB")
- Multiple simultaneous model downloads
