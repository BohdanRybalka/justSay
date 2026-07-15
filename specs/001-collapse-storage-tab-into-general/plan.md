# Spec 001 — Collapse Storage tab into General

> **Status:** done
> **Target release:** vX.Y.Z

## Context / Why

The Settings UI has grown a `Storage` tab (`src/settings/tabs/storage.ts`) whose three
subsections have decayed into low-value clutter: `Transcript History` only shows an entry
count nobody acts on, `Temporary Files` duplicates path/size info nobody needs a whole tab
for, and `Output Directory` is a single input + Browse button that doesn't need its own nav
entry. This is `docs/TODO.md` → Backlog item 2, a follow-on cleanup pass to the same UX
simplification effort as the Storage/Keys/Audio-consolidation backlog item above it. The
outcome is fewer tabs, fewer redundant fields, and the one thing people actually use
(output directory + a way to reclaim temp-file disk space) folded into `General`. This is a
cosmetic/DX cleanup, not a behavior change — no pipeline, STT, or persistence logic is
touched.

## Acceptance Criteria

- `index.html` no longer has a `data-tab="storage"` nav button; the sidebar nav list has 8
  entries instead of 9 (General, Models, Audio, Transcribe, Keys, History, Metrics, Words).
- `src/settings/tabs/storage.ts` is deleted; `src/settings/settings.ts` no longer imports or
  registers `renderStorage`/`storage` in the `tabs` map.
- `src/settings/tabs/general.ts` renders a new `Files` setting-group containing:
  - the output-directory text input + `Browse` button (same debounce-save-on-input and
    Tauri `open()` dialog behavior as the old `storage.ts`, including the sync-folder hint
    and the inline warning/error status line),
  - a `Size` row showing the temp-file size next to a `Clear Temp Files` button (same
    `api.cleanupTemp()` call and "Freed X" / "Failed" feedback as before).
- The old `Transcript History` subsection (Path row, Entries row, `Copy Path` button) is
  gone from the UI entirely — no equivalent is added to General.
- The old `Temporary Files` → `Location` row (raw temp-dir path display) is gone from the
  UI entirely.
- Switching away from the `General` tab while an output-dir debounce timer is pending does
  not leak a timer or throw (destroy function clears it), matching the guarantee the old
  `storage.ts` never actually had to provide (it was a top-level tab, not one that shares a
  destroy lifecycle with unrelated General fields) — General's returned cleanup function
  must clear the debounce timer in addition to resetting `recording`.
- `npx tsc --noEmit` passes (or the project's configured frontend type-check command) with
  no leftover references to `renderStorage` or the deleted DOM ids
  (`history-path-display`, `history-entries-display`, `temp-dir`, `btn-reveal-history`).
- Backend (`backend/app/core/settings_router.py`, `GET /settings/storage`) is unchanged —
  the frontend simply stops reading `history_path`, `history_entries`, and the raw
  `temp_dir` string out of the existing `StorageInfo` response; it keeps reading
  `temp_size_bytes` and calling `POST /settings/cleanup`. No backend test needs to change.

## Recommended approach

Move the surviving DOM/logic from `storage.ts` into `general.ts` verbatim (same element
ids where practical, same debounce constant of 600ms, same error/revert behavior on failed
`saveSettings({ output_dir })`), then delete `storage.ts` and its two registration points.

Concretely, in `src/settings/tabs/general.ts`:
- Add a `Files` `setting-group` (title `Files`, not `Output Directory` — the rename the
  backlog item asks for) positioned as the last group before `About`, since it is the most
  "operational / infrequently touched" group and About is already the fixed bottom anchor.
- Port `output-dir` input + `btn-browse` button + `output-dir-status` hint element,
  `persistOutputDir()`, `showStatus()`/`clearStatus()`, and the `input` debounce listener
  from `storage.ts` lines 9–18, 51–53, 61–114 essentially unchanged.
- Port the `temp-size` value + `btn-cleanup` button from `storage.ts` lines 41–48, 130–144,
  but drop the `temp-dir` (`Location`) row entirely. Rename the local loader from
  `loadStorageInfo` to something scoped to `general.ts` (e.g. `loadFilesInfo`) since it now
  only needs to populate `tempSize` — it still calls `api.getStorageInfo()` (the endpoint is
  untouched) and reads only `temp_size_bytes` off the response, ignoring the other three
  fields.
- Do NOT port `history-path-display`, `history-entries-display`, `btn-reveal-history`, or
  the `Transcript History` group at all — this subsection is deleted, not moved.
- Extend `general.ts`'s returned cleanup closure (currently `() => { recording = false; }`)
  to also clear the ported debounce `setTimeout` handle, since General's destroy function is
  now responsible for a timer that `storage.ts` (a self-contained top-level tab) didn't have
  to share a lifecycle with.

In `src/settings/settings.ts`: delete the `import { renderStorage } from "./tabs/storage";`
line and the `storage: renderStorage,` entry in the `tabs` record.

In `index.html`: delete the `<li><button class="nav-btn" data-tab="storage">Storage</button></li>`
line.

Delete `src/settings/tabs/storage.ts` outright.

No ADR — this is a straightforward UI-consolidation move with no architectural decision to
record (no new pattern, no cross-cutting tradeoff); it follows the same pattern the sibling
backlog item (Keys → General) already established.

## Files to modify

- `src/settings/tabs/general.ts` — add the `Files` setting-group (output-dir input/Browse,
  temp size + Clear Temp Files), port the debounce-save and cleanup logic, extend the
  destroy closure to clear the debounce timer.
- `src/settings/settings.ts` — remove `renderStorage` import and its `tabs` map entry.
- `index.html` — remove the `data-tab="storage"` nav button.
- `src/settings/tabs/storage.ts` — delete file.

## Risks

- The output-dir debounce/persist logic (600ms `setTimeout`, revert-on-error) is being
  copy-moved by hand rather than imported as a shared function; a future edit to one copy's
  behavior (there's now only one copy, but reviewers should double check no second copy
  survives anywhere, e.g. in a stale build artifact) won't apply itself elsewhere — low risk
  here since the old file is deleted in the same change, but worth a diff-level sanity check.
- `general.ts` already owns one `setTimeout`-based interaction pattern-free of debounce
  (shortcut recording uses a document-level keydown listener, not a timer) — adding the
  first debounce timer to this file's destroy lifecycle is new surface area for the destroy
  function; if a future field is added to General with its own timer and a developer forgets
  to extend the same destroy closure, that timer will leak silently (no test currently
  exercises tab-switch-while-pending-debounce for General).
- `api.getStorageInfo()` continues to fetch four fields over the wire when only one
  (`temp_size_bytes`) is used client-side; this is intentionally deferred (see Cuts below)
  but is a real, if minor, residual inefficiency and a piece of dead-code smell in the
  backend response contract until that follow-up lands.

## Cuts deferred to a future spec

| What | Why | Trigger to reinstate |
| --- | --- | --- |
| Trimming `StorageInfo` (backend `settings_router.py`) down to just `temp_size_bytes`, removing `temp_dir`, `output_dir`, `history_path`, `history_entries` from the response model and `_mask_home` usage tied to them | Backlog item is scoped to the frontend Settings UI; touching the backend response model also means updating/deleting `backend/tests/test_api.py::test_storage_info_routes_output_dir_through_mask` and re-justifying whether `_mask_home` (still unit-tested directly) is dead code — that's a separate, backend-focused cleanup decision, not a UI tab collapse | Next time `StorageInfo` gains or loses a consumer, or when a dedicated backend dead-code pass is scheduled |
| A dedicated "Copy history file path" affordance somewhere else in the UI | The backlog explicitly calls the `Transcript History` subsection (which included the Copy Path button) "not useful" and asks for full deletion; if a user later asks for a quick way to locate the history file, that's a new, deliberately-scoped feature request, not a restoration of the deleted subsection | A concrete user request for "where is my history file" surfaces |
| Shared/extracted debounce-save helper (e.g. a `useDebouncedSetting` utility) instead of the inline `setTimeout` pattern duplicated across settings tabs (`audio.ts`, now `general.ts`) | Out of scope for a tab-collapse; extracting a shared helper is a separate refactor with its own review surface | A third tab needs the same debounce-save pattern, making the duplication a real (not just theoretical) DRY violation |

## Deviations

## Review history

### Review — iteration 1 (2026-07-15)
- reviewer verdict: approved
- No RED. Two non-blocking YELLOW notes: (1) no test/manual-QA note added for the tab-switch-while-pending-debounce risk already flagged in this plan's Risks section; (2) `api.getStorageInfo()` still fetches 3 unused fields over the wire (intentionally deferred per Cuts).
