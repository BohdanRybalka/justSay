# Plan 001: MVP — Instant Prompt

## Goal
Minimal working pipeline: press hotkey → speak → text appears in clipboard, cleaned and grammar-corrected.

## Why start here
- Most contained scenario (no Obsidian integration, no system audio capture)
- Immediate daily value for the user
- Establishes the full vertical slice: UI → Audio → STT → LLM → Clipboard

---

## Phase 1: Project Skeleton

### 1.1 Python Backend (FastAPI)
- [ ] Initialize Python project structure (`backend/`)
- [ ] FastAPI app with health endpoint
- [ ] Config layer: settings.yaml or .env for API keys and model selection
- [ ] Abstract `STTProvider` interface (Cloud/Local contract)
- [ ] Abstract `LLMProvider` interface (Cloud/Local contract)

### 1.2 Tauri Frontend
- [ ] Initialize Tauri v2 project (`src-tauri/` + frontend in `src/`)
- [ ] System tray with basic menu (Start/Stop, Settings, Quit)
- [ ] Global hotkey registration (configurable)
- [ ] IPC bridge to Python backend (sidecar or HTTP localhost)

### 1.3 Integration
- [ ] Frontend starts Python backend as sidecar process
- [ ] Health check: frontend verifies backend is alive
- [ ] Basic error handling and status display

---

## Phase 2: Audio Capture

### 2.1 Microphone Recording
- [ ] Python module: capture microphone audio (PyAudio or sounddevice)
- [ ] Start/stop recording via API endpoint
- [ ] Save audio to temp WAV file
- [ ] Audio level indicator data (optional, for UI feedback)

---

## Phase 3: STT (Speech-to-Text)

### 3.1 Cloud Mode (API-First)
- [ ] Implement `CloudSTTProvider` using Gemini 2.5 Flash Native Audio API
- [ ] Send audio file → receive transcription
- [ ] Handle errors, timeouts, rate limits

### 3.2 Local Mode (Privacy-First)
- [ ] Implement `LocalSTTProvider` using faster-whisper
- [ ] Auto-download model on first use
- [ ] GPU detection and fallback to CPU

### 3.3 Provider Selection
- [ ] Config switch: `stt.mode: cloud | local`
- [ ] Runtime switching via API endpoint

---

## Phase 4: Text Processing (LLM)

### 4.1 Cleanup Pipeline
- [ ] System prompt: remove filler words, fix grammar, preserve meaning
- [ ] Cloud mode: Groq API (Llama 4 Scout) for fast inference
- [ ] Local mode: Ollama (Gemma 3 4B)

### 4.2 Provider Selection
- [ ] Config switch: `llm.mode: cloud | local`
- [ ] Runtime switching via API endpoint

---

## Phase 5: Clipboard & UX Polish

- [ ] Copy final text to system clipboard (backend or frontend)
- [ ] Toast/notification: "Text copied!" with preview
- [ ] Status indicator in system tray (idle → recording → processing → done)
- [ ] Hotkey workflow: hold to record, release to process (push-to-talk)

---

## Architecture Result

```
[Global Hotkey] → Tauri UI
       ↓
[Start Recording] → Python Backend → Microphone
       ↓
[Stop Recording] → Audio File (temp)
       ↓
[STT Provider] → Raw Text
       ↓
[LLM Provider] → Cleaned Text
       ↓
[Clipboard] → User pastes anywhere
```

## Tech Decisions (Resolved)

1. **Tauri sidecar + HTTP localhost** — Tauri manages Python process lifecycle, communication via HTTP on 127.0.0.1. Best of both: clean lifecycle + debuggable API.
2. **sounddevice** — better API, good Windows support, async-friendly
3. **Push-to-talk** — hold `Ctrl+Alt+V` to record, release to process (configurable)
4. **WAV** — lossless, simple, convert to OPUS only for cloud upload if needed

## Out of Scope (for MVP)
- System audio capture (WASAPI loopback) — that's Project Memory
- Obsidian integration
- Structured output templates
- UI settings panel (config via file is fine for MVP)
- Auto-update mechanism
