# Plan 003: Settings Panel (Admin Dashboard)

## Goal

Replace the current basic settings window with a full admin panel featuring tabbed navigation, configuration management, and audio controls.

## Architecture Decision: Two-Layer Config

```
┌─────────────────────────────────────┐
│ Layer 1: Secrets (.env)             │  ← API keys only
│ Read once at startup                │  ← Never written by app
│ .gitignore'd                        │
├─────────────────────────────────────┤
│ Layer 2: User Settings              │  ← Language, paths, model choices, modes
│ ~/.justsay/settings.json            │  ← Read/write at runtime
│ JSON with Pydantic validation       │  ← Auto-created with defaults
└─────────────────────────────────────┘
```

## Window Layout

```
┌─────────────────────────────────────────────────────┐
│  JustSay Settings                            [─][×] │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│  General │   < Active tab content >                 │
│  Models  │                                          │
│  Audio   │                                          │
│  Storage │                                          │
│          │                                          │
│          │                                          │
│          │                                          │
│──────────│──────────────────────────────────────────│
│  v0.1.0  │                          Backend: ● OK   │
└──────────┴──────────────────────────────────────────┘
```

## Tabs (4 tabs, no API Keys tab — keys stay in .env)

### 1. General
- **Language**: Dictation language selector (uk, en, de, etc.)
- **Global shortcut**: Display current hotkey (Ctrl+Alt+V), read-only for now

### 2. Models
- **STT Mode**: Cloud / Local toggle
  - Cloud: Model name (gemini-2.5-flash), read-only info
  - Local: Model selector (large-v3, medium, small, base), device (auto/cpu/cuda)
- **LLM Mode**: Cloud / Local toggle
  - Cloud: Model name (llama-4-scout), read-only info
  - Local: Ollama host, model name
- **Ollama status**: Auto-check connectivity when Local selected

### 3. Audio
- **Microphone test**: Record button + live level meter (dBFS bar)
- **Sample rate**: Display current (16000 Hz), read-only
- **Max recording duration**: Editable (default: 5 min)
- **Quick dictation**: Record → Transcribe → Show result
- **Conflict guard**: Disable record if widget is already recording

### 4. Storage
- **Output directory**: Path input + folder picker button (Tauri dialog)
  - Default: `~/.justsay/output/`
- **Temp directory**: Display path (`~/.justsay/tmp/`), read-only
- **Cleanup**: Button to clear temp files, show current size

## Technical Approach

### Frontend
- **Single page with client-side tab routing** (vanilla TS, no framework)
- Each tab = a TS module exporting `render(container)` and `destroy()` functions
- File structure:
  ```
  src/settings/
    settings.ts        — Tab router, init, settings state
    tabs/
      general.ts       — General tab
      models.ts        — Models configuration tab
      audio.ts         — Audio test & config tab
      storage.ts       — Storage paths tab
    components/
      toggle.ts        — Cloud/Local toggle component
      level-meter.ts   — Audio level visualization
    settings.css       — All settings styles
  ```

### Backend
- New `UserSettings` model in `backend/app/core/user_settings.py`
- New endpoints:
  ```
  GET  /settings          → current UserSettings
  PUT  /settings          → update UserSettings (merge)
  GET  /settings/storage  → {temp_dir, temp_size_bytes, output_dir}
  POST /settings/cleanup  → delete tmp files, return freed bytes
  ```
- Existing endpoints stay: PUT /stt/mode, PUT /llm/mode (proxy to UserSettings)

### Tauri
- Window size: 680x480 (up from 500x400)
- Add `tauri-plugin-dialog` for folder picker

## Implementation Phases

### Phase 1: UserSettings backend + Shell & Navigation
- `UserSettings` Pydantic model with load/save to `~/.justsay/settings.json`
- `/settings` GET/PUT endpoints
- Tab layout with sidebar navigation (HTML/CSS/TS)
- Tab routing (show/hide content)
- Backend status indicator in footer
- Replace current index.html/main.ts/styles.css

### Phase 2: General + Models tabs
- Language selector with backend sync
- Cloud/Local toggles synced to UserSettings
- Model info display
- Ollama status check

### Phase 3: Audio tab
- Record/stop button with level meter (poll /audio/status)
- Quick dictation flow
- Audio settings display
- Conflict guard (check if widget is recording)

### Phase 4: Storage tab
- Storage info endpoint
- Directory picker via Tauri dialog
- Temp cleanup button

## Design Principles
- Dark theme (#1a1a2e base, consistent with widget)
- Minimal, clean layout
- Auto-save with debounce (500ms) + validation before save
- Status indicators for external dependencies (backend, Ollama)
