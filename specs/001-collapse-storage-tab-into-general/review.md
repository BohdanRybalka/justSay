# Review — specs/001-collapse-storage-tab-into-general

## RED — Critical Issues

Немає.

## YELLOW — Concерns

- `src/settings/tabs/general.ts:158-161` — debounce-таймер (`debounceTimer`) є звичайним module-scope-у-функції `let`, який очищається в поверненому cleanup-closure (`general.ts:319-322`). Це коректно закриває AC про відсутність витоку таймера при перемиканні вкладок. Але сам АС явно позначений у плані як «no test currently exercises tab-switch-while-pending-debounce for General» (Risks, plan.md:107) — це залишається неперевіреним вручну/тестами і в цьому діффі: жодного нового тесту або навіть ручної QA-нотатки в `Deviations`/`Review history` плану не додано. Не блокує (сама плата за ризик вже документована автором плану), але вартий згадки як залишковий пробіл.
- `src/settings/tabs/general.ts:325-332` — `loadFilesInfo` продовжує викликати `api.getStorageInfo()`, який під капотом усе ще повертає `history_path`, `history_entries`, `temp_dir` по мережі (бекенд навмисно не займали — це задокументовано в Cuts). Підтверджую, що це відповідає плану, не new regression, але фактично зайвий проходить по дроту payload, який тепер повністю ігнорується клієнтом — залишковий tech debt, як і зазначено в плані.

## GREEN — Improvements

- Перенесення `output-dir`/`btn-browse`/`temp-size`/`btn-cleanup` логіки виглядає посимвольно ідентичним до видаленого `storage.ts` (та ж 600ms debounce-константа, той самий revert-on-error, той самий `formatBytes`/`escapeHtml`) — точна відповідність «Recommended approach» з плану.
- Розташування нової групи `Files` — прямо перед `About` (`general.ts:54-71`), як і вимагалося в плані.

## Перевірка Acceptance Criteria

- `index.html`: nav-список тепер має 8 пунктів (General, Models, Audio, Transcribe, Keys, History, Metrics, Words), `data-tab="storage"` видалено — підтверджено читанням файлу.
- `src/settings/tabs/storage.ts` видалено; `src/settings/settings.ts` більше не імпортує/не реєструє `renderStorage`/`storage` — підтверджено діффом.
- `general.ts` містить нову групу `Files` з input `#output-dir` + `Browse`, sync-folder hint, `#output-dir-status`, та рядок `Size` (`#temp-size`) + `Clear Temp Files` (`#btn-cleanup`) — увесь код логіки (`persistOutputDir`, `showStatus`/`clearStatus`, debounce-listener, `loadFilesInfo`, cleanup-handler) присутній і поведінково ідентичний старому `storage.ts`.
- `Transcript History` (Path/Entries/Copy Path) повністю відсутня в новому коді — не перенесена, як і вимагалося.
- `Temporary Files → Location` (`#temp-dir`) видалено повністю — залишився лише `Size`.
- Cleanup-closure General (`general.ts:319-322`) тепер очищає `debounceTimer` на додачу до `recording = false` — відповідає вимозі.
- `npx tsc --noEmit` пройшов без помилок (перевірено безпосередньо); grep по всьому `src/` не знайшов жодних залишкових посилань на `renderStorage`, `history-path-display`, `history-entries-display`, `temp-dir`, `btn-reveal-history` (єдине згадування `temp-dir`-подібного рядка — у `backend/app/core/settings_router.py`, що очікувано, бекенд не займали).
- Бекенд (`settings_router.py`, `GET /settings/storage`) не займали — підтверджено, у git diff цього файлу немає, і scope діффу такий, що backend файл навіть не входив до списку `Files to modify`.

## Verdict: approved
