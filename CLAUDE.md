# JustSay

Personal voice-processing operating system. Converts chaotic audio streams (thoughts, meetings, prompts) into structured knowledge with full manual control over privacy via hybrid Cloud/Local architecture.

## Business Scenarios

1. **Instant Prompt** — Voice-to-clipboard dictation with minimal latency, auto-cleanup of filler words and grammar correction
2. **Project Memory** — Meeting archival into Obsidian with summaries, decisions, and action items
3. **Structured Thoughts** — Stream-of-consciousness to structured data via customizable templates

## Architecture

```
┌─────────────────────────────────────┐
│  UI/Shell Layer (TypeScript/Tauri)  │
│  - System tray, global hotkeys      │
│  - Obsidian filesystem integration   │
│  - Model switcher [Cloud/Local]      │
└──────────────┬──────────────────────┘
               │ HTTP/IPC
┌──────────────▼──────────────────────┐
│  AI Engine Layer (Python/FastAPI)    │
│  - Audio capture (WASAPI Loopback)   │
│  - Model orchestration (API/Ollama)  │
│  - Contract: Audio In → Text Out     │
└─────────────────────────────────────┘
```

### Model Strategy (Manual Cloud/Local Switch)

| Task            | Cloud (API-First)              | Local (Privacy-First)              |
|-----------------|-------------------------------|------------------------------------|
| STT             | Gemini 2.5 Flash Native Audio | whisper.cpp large-v3-turbo (Metal) |
| Fast Reasoning  | Llama 4 Scout (Groq API)      | Qwen3 1.7B (Ollama/Metal)         |
| Deep Analysis   | Gemma 4 31B (Google API)      | Qwen3 4B (Ollama/Metal)           |

**Local mode target platform**: macOS Apple Silicon (M1+ 8GB unified memory)
**Performance targets**: Short audio (<40s) → 1-2.5s, Long audio (40-150s) → up to 10s
**Windows**: Cloud mode only (AMD GPUs lack CUDA/ROCm support for local AI)

## Development Rules

### Language

- Communication with the user: **Ukrainian**
- Code, commits, code comments: **English**
- Project documentation filenames: **English**

### Workflow — MANDATORY sequence for every feature/fix

1. **Plan** — create `docs/plans/<feature-name>.md` BEFORE writing any code
2. **Architect review** — run `/architect` on the plan; address feedback before proceeding
3. **Implement** — write code per Architecture Principles and Code Style
4. **QA review** — run `/qa` on the result; fix any critical issues
5. **Document** — update `docs/TODO.md`, add entry to `docs/release-notes/`

Never start step 3 before steps 1–2 are done.
Never call a feature complete before steps 4–5 are done.

Artifacts:
- Tasks: `docs/TODO.md`
- Plans: `docs/plans/<feature-name>.md`
- Release Notes: `docs/release-notes/vX.X.X.md`

### Architecture Principles

- Backend (Python) is **model-agnostic** — executes contract Audio In → Text Out
- Frontend (TypeScript/Tauri) is **lightweight** — system events and state visualization only
- Model settings live in a **separate configuration layer**
- **Modular approach**: each component has a single clear responsibility
- Local mode = **zero data leakage** — no bytes leave the local machine
- Cloud mode = **BYOK** (Bring Your Own Key) only

### Code Style

- TypeScript: strict mode, ES modules
- Python: type hints, async/await where applicable
- Tests required for business logic

### Don'ts

- Don't add features without user request
- Don't over-engineer for hypothetical futures
- Don't commit without explicit user request
- Don't ignore privacy constraints in Local mode
- Don't create vendor lock-in on any specific model

### Available Commands

- `/qa` — Critical QA review of code, plans, ideas, tasks
- `/architect` — Architectural review from the product vision perspective
