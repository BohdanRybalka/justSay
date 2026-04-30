# Plan 002: Dynamic Island Widget + Settings Window

## Goal
Replace single window with two-window architecture:
1. **Widget** — small always-on-top semi-transparent overlay for recording
2. **Settings** — full window for configuration, file upload, history

## Phase A: Fix Click Toggle (Bug Fix)

- [ ] Change button behavior: click = start, click again = stop (not mousedown/mouseup)
- [ ] Keep Ctrl+Alt+V as push-to-talk (hold to record, release to stop)

## Phase B: Widget Window

- [ ] Create Tauri widget window config: ~300x60px, always_on_top, transparent, no decorations
- [ ] Widget HTML/CSS: pill-shaped, semi-transparent dark, shows status
- [ ] Widget states: idle (minimal), recording (pulsing red), processing (spinner), done (green flash)
- [ ] Widget shows: duration during recording, result preview after done
- [ ] Click on widget toggles recording
- [ ] Draggable by user to any screen position
- [ ] Widget is always visible — no window can cover it

## Phase C: Settings Window

- [ ] Rename existing main window to "settings"
- [ ] Open from tray menu "Settings" (not auto-shown)
- [ ] Tabs/sections: General, Providers (Cloud/Local), Audio, History
- [ ] File upload for transcription in Settings
- [ ] Language selector

## Tech Decisions

- Tauri supports multiple windows with different configs
- `always_on_top: true` + `transparent: true` + `decorations: false` for widget
- Widget and settings share the same Tauri app, different HTML entry points
- Widget communicates with backend via same fetch API

## Implementation Order

1. Phase A (5 min) — fix click toggle
2. Phase B (main work) — widget window
3. Phase C (later) — settings window expansion
