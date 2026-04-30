# Plan: Hybrid STT Pipeline — Smart Routing + LLM Removal

**Version:** 2.0 (post QA + Architect review)

## Goal
Replace the current 2-step pipeline (STT -> LLM) with a smart single-step pipeline:
- **Short audio + normal style:** Groq Whisper -> raw text -> done (fastest path)
- **Long audio OR ai_prompt style:** Gemini Flash -> transcribe (+ structure if ai_prompt) -> done
- **Local:** faster-whisper only, remove Ollama from default pipeline

## Current Architecture
```
Audio -> STT (Gemini/Whisper) -> raw_text -> LLM (Groq Llama/Ollama) -> cleaned_text -> clipboard
```
Files: `pipeline/service.py`, `stt/cloud.py`, `stt/local.py`, `llm/cloud.py`, `llm/local.py`

## Target Architecture
```
Audio -> detect_duration -> get_routed_provider() -> provider.transcribe() -> text -> clipboard
                                    |
                   +----------------+----------------+
                   |                |                |
            Groq Whisper     Gemini Flash     faster-whisper
            (short+normal)   (long OR ai)     (all local)
```

**Key design principle:** Pipeline stays "dumb" — one `stt.transcribe()` call.
All routing logic lives in `stt/__init__.py` / `stt/routing.py`, NOT in pipeline.

---

## Task 1: Add Groq Whisper STT Provider

### What
New `GroqWhisperSTTProvider` class in `app/stt/groq_whisper.py`.
Implements `STTProvider` interface (no new methods).

### Implementation Details
- Use existing `groq` SDK (already in `cloud` deps) — `client.audio.transcriptions.create()`
- Model: `whisper-large-v3-turbo` (configurable via `STTSettings.groq_whisper_model`)
- Input: audio file path. Opens file as binary, sends to Groq API.
- Output: transcribed text string
- `response_format="text"` for speed (no JSON parsing overhead)
- `cleanup()`: no-op (HTTP client is stateless), but set `self._client = None` for consistency

### Config Changes (`stt/config.py`)
```python
# Add to STTSettings:
groq_api_key: str = ""
groq_whisper_model: str = "whisper-large-v3-turbo"
```

### Edge Cases & Limitations
1. **File size limit:** Groq free tier = 25 MB. Live recording 30s WAV 16kHz mono = ~960 KB — well under. For file uploads, the existing `MAX_UPLOAD_SIZE=25MB` aligns.
2. **Format:** Groq accepts WAV, MP3, FLAC, OGG. Recorder outputs WAV — OK. Note: `.webm` is in `ALLOWED_EXTENSIONS` but NOT supported by Groq — need per-provider validation (see Task 2).
3. **Rate limits (free):** 2000 RPD. Log clear error on HTTP 429 with message "Groq rate limit exceeded, try again later or switch to Gemini."
4. **Timeout:** Set SDK timeout to 10s (generous for short audio).
5. **Language:** Groq Whisper `language` param uses ISO-639-1 ("uk", "en") — matches our codes.
6. **API key:** Separate `JUSTSAY_STT_GROQ_API_KEY`. NO automatic fallback from `LLM_GROQ_API_KEY` in code (cross-module coupling). Instead: document that user can set same key for both. Resolve in `sync_to_runtime()` if needed.
7. **Error handling:** On Groq failure, raise exception. Pipeline returns error to user. No auto-fallback to Gemini (too complex, user can switch manually).

### Tests
- Mock `groq.Groq` client, test `transcribe()` returns text
- Test HTTP 429 handling (clear error message)
- Test missing API key error
- Test `cleanup()` resets client

---

## Task 2: Smart Routing via STT Factory (NOT in pipeline)

### What
Add routing function in `stt/__init__.py` that selects provider based on:
- `mode` (cloud/local)
- `audio_duration` (short/long)
- `style` (normal/ai_prompt)

Pipeline calls ONE function and gets the right provider back.

### Architecture Decision (from Architect review)
**Routing lives in STT module, NOT in pipeline.** Pipeline remains a dumb orchestrator.

```python
# stt/__init__.py
def get_routed_provider(
    settings: STTSettings,
    audio_duration: float | None = None,
    style: str = "normal",
) -> STTProvider:
    """Select provider based on mode + duration + style."""
    if settings.mode == ProviderMode.LOCAL:
        return _get_local_provider(settings)

    # Cloud routing:
    # ai_prompt always goes to Gemini (needs structuring capability)
    if style == "ai_prompt":
        return _get_gemini_provider(settings, style=style)

    # normal style: route by duration
    if audio_duration is not None and audio_duration <= settings.cloud_routing_threshold:
        return _get_groq_provider(settings)

    # Long audio or unknown duration -> Gemini
    return _get_gemini_provider(settings, style=style)
```

**Why `style == "ai_prompt"` always goes to Gemini:**
If user explicitly chose structured output, they expect structuring even for 5s of audio.
Groq Whisper can't structure. Ignoring user's style choice would be surprising.

### Pipeline Becomes
```python
# pipeline/service.py
async def process_audio(audio_path, language, style, copy_to_clipboard, audio_duration=None):
    duration = audio_duration or detect_duration(audio_path)
    stt = get_routed_provider(settings.stt, audio_duration=duration, style=style)
    text = await stt.transcribe(audio_path, language=language)

    # No LLM step
    # clipboard + history...
```

### Duration Detection

#### CRITICAL FIX: `recorder.duration_seconds` returns 0.0 after `stop()`
`MicrophoneRecorder.duration_seconds` returns `0.0` when `_recording == False`.
`stop()` sets `_recording = False` BEFORE returning path.
So by the time pipeline runs, duration is already 0.0.

**Fix:** Modify `recorder.stop()` to return `(path, duration)` tuple, or save duration in `stop()`:
```python
# recorder.py — modify stop()
async def stop(self) -> Path:
    with self._lock:
        if not self._recording or self._stream is None:
            raise RuntimeError("Not recording")
        self._recording = False
        self._final_duration = time.monotonic() - self._start_time  # SAVE before clearing

# Add property:
@property
def last_duration_seconds(self) -> float:
    """Duration of the last completed recording."""
    return getattr(self, '_final_duration', 0.0)
```

Then in `pipeline/router.py`:
```python
audio_path = await recorder.stop()
duration = recorder.last_duration_seconds
result = await process_audio(audio_path, ..., audio_duration=duration)
```

#### File upload duration detection
Use `soundfile.info(path).duration` — `soundfile` is already in `audio` extras.
```python
# pipeline/utils.py or audio/utils.py
def detect_duration(audio_path: Path) -> float | None:
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        return None  # unknown duration -> route to Gemini
```

### Provider Caching (Dict-based, Thread-safe)

Replace single `_cached_provider` with dict keyed by provider class:

```python
import threading

_cache_lock = threading.Lock()
_providers: dict[type, STTProvider] = {}

def _get_groq_provider(settings: STTSettings) -> STTProvider:
    from app.stt.groq_whisper import GroqWhisperSTTProvider
    return _get_or_create(GroqWhisperSTTProvider, settings)

def _get_gemini_provider(settings: STTSettings, style: str = "normal") -> STTProvider:
    from app.stt.cloud import GeminiSTTProvider
    # Style is passed via settings or constructor, NOT via transcribe()
    return _get_or_create(GeminiSTTProvider, settings, style=style)

def _get_local_provider(settings: STTSettings) -> STTProvider:
    from app.stt.local import LocalSTTProvider
    return _get_or_create(LocalSTTProvider, settings)

def _get_or_create(cls, settings, **kwargs) -> STTProvider:
    with _cache_lock:
        if cls in _providers:
            return _providers[cls]
        provider = cls(settings, **kwargs)
        _providers[cls] = provider
        return provider

def clear_cache():
    with _cache_lock:
        for p in _providers.values():
            p.cleanup()
        _providers.clear()
```

### Edge Cases & Limitations
1. **Threshold boundary:** `<=` for short (Groq). Exactly 30s -> Groq. Consistent.
2. **Duration detection failure:** Returns `None` -> routes to Gemini (safe default for unknown-length files).
3. **File size vs duration mismatch:** A 20MB file that's only 10s long (high bitrate) — duration says "short" -> routes to Groq. But Groq accepts up to 25MB, so 20MB is fine. Only files >25MB are rejected by router's `MAX_UPLOAD_SIZE` before reaching pipeline.
4. **Threshold validation:** `cloud_routing_threshold` must be `> 0`. Upper bound: not enforced in code, but document that >120s means most live recordings go to Groq. Add pydantic `@field_validator`.
5. **Concurrency:** Thread-safe dict cache with `_cache_lock` handles concurrent requests using different providers.
6. **Cache invalidation on config change:** `sync_to_runtime()` calls `clear_cache()` when settings change — both providers get cleaned up. New ones created lazily on next request.
7. **Gemini style caching:** If user switches from `normal` to `ai_prompt`, the cached Gemini provider has the old style. Fix: either don't cache style in provider (pass per-call), or include style in cache key. **Decision:** Style changes prompt, not the provider. Store style in settings and read it at transcribe-time, not at init-time (see Task 3).

### Fix: `/process-file` missing `style` parameter
Current `process_file()` endpoint doesn't accept `style`. Add it:
```python
@router.post("/process-file", response_model=DictateResponse)
async def process_file(file: UploadFile, language: str = "uk", style: str = "normal", ...):
```

### Tests
- Test routing: short+normal -> Groq, short+ai_prompt -> Gemini, long+any -> Gemini, local -> faster-whisper
- Test threshold boundary (exactly 30s)
- Test duration auto-detection from WAV file
- Test `detect_duration` returns None for corrupt file
- Test `recorder.last_duration_seconds` after stop()
- Test concurrent requests with different providers (thread safety)
- Test `clear_cache()` cleans up all providers

---

## Task 3: Gemini Style-Aware Transcription (NO new interface method)

### What
Modify `CloudSTTProvider` (rename to `GeminiSTTProvider`) to accept `style` and build the appropriate prompt. **NO `transcribe_with_style()` method.** The `transcribe()` interface stays unchanged.

### Architecture Decision (from Architect review)
**`STTProvider.transcribe(path, language)` stays clean.** Style is NOT part of the STT interface.
Style influences the **prompt**, not the method signature. Gemini provider reads style at transcribe-time from settings/config.

### Implementation
```python
class GeminiSTTProvider(STTProvider):
    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._client = None

    async def transcribe(self, audio_path: Path, language: str = "uk") -> str:
        # Read current style from user settings (not cached at init)
        from app.core.user_settings import get_user_settings
        style = get_user_settings().transcription_style

        prompt = self._build_prompt(language, style)
        # ... send audio + prompt to Gemini
```

**Alternative:** Pass `style` via `STTSettings` and let pipeline set it before calling transcribe. But reading from `UserSettings` is simpler and consistent with how settings flow elsewhere.

**Best option:** Pipeline passes `style` into `get_routed_provider()`, routing function stores it in a lightweight context that Gemini provider reads. This avoids coupling to UserSettings and keeps testability.

**Final decision:** Add `style` parameter to `STTSettings` as a runtime-mutable field:
```python
class STTSettings:
    # ... existing fields ...
    _current_style: str = "normal"  # set by pipeline before routing
```

Or simpler: add `style` to `transcribe()` as **optional kwarg with default**:
```python
class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs) -> str:
        ...
```

Gemini reads `kwargs.get("style", "normal")`. Other providers ignore it. This is **minimal interface change, fully backwards compatible**.

### Prompts

#### Normal style (existing, no change)
```
Transcribe this audio faithfully.
The primary language is {language}.
The speaker may use words from other languages — write them in their original form.
Include natural punctuation based on speech intonation.
Output ONLY the transcription text, nothing else.
```

#### ai_prompt style (new combined prompt)
```
Transcribe this audio and structure the output as a professional document.
The primary language is {language}.
The speaker may use words from other languages — write them in their original form.

Instructions:
1. Transcribe faithfully, removing speech disfluencies (hesitation, filler words, repeated words).
2. Fix grammar, spelling, punctuation appropriate for {language}.
3. Analyze speaker's intent and structure appropriately:
   - Task or request -> action items with context
   - Idea or concept -> key points
   - Problem description -> problem statement + expected behavior
   - List of items -> bulleted list
4. Add headings and lists where they improve clarity.
5. Preserve proper nouns, brand names, technical terms exactly.
6. Do not add information not present in the audio.

Output ONLY the structured text.
```

### Edge Cases
1. **Empty/silent audio:** Gemini may return empty or "No speech detected." Check: if result is empty or starts with common refusal patterns ("I cannot", "No audio"), return empty string.
2. **Gemini preamble:** "Here is the transcription:" — prompt says "Output ONLY". If it still happens, strip first line if it matches pattern.
3. **raw_text vs cleaned_text:** When Gemini does both STT+structuring, `raw_text` equals `cleaned_text`. This is a known limitation. Document in API response description.
4. **Prompt stored in `pipeline/prompts.py`:** Move Gemini-specific prompts there or keep in `stt/cloud.py`. Decision: keep transcription prompts in `stt/cloud.py` (they're provider-specific). Move generic prompt templates to `pipeline/prompts.py` only if shared.

### Tests
- Test prompt generation for normal vs ai_prompt style
- Test that `**kwargs` with `style` param works for Gemini
- Test that Groq/local providers ignore `style` kwarg
- Test empty audio handling

---

## Task 4: Remove LLM from Pipeline

### What
- Remove LLM call from `pipeline/service.py`
- Keep `llm/` module intact for standalone `/llm/process` endpoint

### Implementation

#### `pipeline/service.py` — new version
```python
async def process_audio(
    audio_path: Path,
    language: str = "uk",
    style: str = "normal",
    copy_to_clipboard: bool = True,
    audio_duration: float | None = None,
) -> ProcessingResult:
    start = time.perf_counter()

    # Detect duration if not provided
    if audio_duration is None:
        audio_duration = detect_duration(audio_path)

    # Single STT step — routing handled by factory
    stt = get_routed_provider(settings.stt, audio_duration=audio_duration, style=style)
    text = await stt.transcribe(audio_path, language=language, style=style)

    log.info("Pipeline route: %s, duration=%.1fs", stt.model_name, audio_duration or -1)

    # Clipboard
    copied = False
    if copy_to_clipboard and text:
        try:
            pyperclip.copy(text)
            copied = True
        except Exception as e:
            log.warning("Clipboard copy failed: %s", e)

    duration_ms = int((time.perf_counter() - start) * 1000)

    # History
    try:
        save_entry(raw_text=text, cleaned_text=text, duration_ms=duration_ms, language=language, style=style)
    except Exception as e:
        log.warning("Failed to save history: %s", e)

    return ProcessingResult(raw_text=text, cleaned_text=text, duration_ms=duration_ms, copied_to_clipboard=copied)
```

#### Removed imports
- `from app.llm import get_llm_provider` — REMOVE
- `from app.pipeline.prompts import get_system_prompt` — REMOVE (prompts now in STT provider)

#### What stays
- `llm/` module — intact
- `/llm/process` endpoint — works standalone
- `/llm/mode` — works

### sync_to_runtime() update
Add new fields sync:
```python
# In sync_to_runtime():
settings.stt.cloud_routing_threshold = us.cloud_routing_threshold
# groq_api_key is from .env, not user settings — no sync needed
```

### Existing test migration
Tests that mock LLM in pipeline will break. Update:
- Remove LLM mocks from pipeline tests
- Add routing-based tests instead
- Keep LLM tests for standalone `/llm/process` endpoint

### Edge Cases
1. **API backwards compatible:** `raw_text == cleaned_text` when LLM is removed. No field removal.
2. **Empty text to clipboard:** If STT returns empty string, don't copy to clipboard (already handled by `if text:` check).
3. **Logging:** Log provider name + duration for debugging which route was taken.

### Tests
- Test full pipeline (cloud short): records -> Groq -> clipboard
- Test full pipeline (cloud long): records -> Gemini -> clipboard
- Test full pipeline (local): records -> faster-whisper -> clipboard
- Test `/llm/process` still works independently
- Test error in STT returns error (not empty clipboard)

---

## Task 5: Remove Ollama from Local Default

### What
- Remove `ollama` from `local` extras in `pyproject.toml`
- Keep `llm/local.py` code (lazy imports, won't crash)
- Pipeline no longer calls LLM — so Ollama is dead code for default flow

### Implementation

#### `pyproject.toml`
```toml
[project.optional-dependencies]
local = [
    "faster-whisper>=1.1.0",
    # ollama removed — not needed for default pipeline
]
local-llm = [
    "ollama>=0.4.0",  # optional, for standalone /llm/process use
]
```

#### `UserSettings` — keep fields, no breakage
`ollama_host`, `ollama_model` fields stay. They're just unused by default pipeline.

#### LLM router endpoints (`llm/router.py`)
- `/llm/local/status` — keep, but check if ollama is installed first
- Add try/except around ollama imports in these endpoints with clear error: "Ollama not installed. Run: pip install justsay-backend[local-llm]"

### Edge Cases
1. **Existing settings.json:** Fields stay, defaults used. No migration needed.
2. **Import safety:** `llm/local.py` uses lazy `from ollama import Client` inside `_get_client()`. If package missing, error is raised only when someone explicitly calls `/llm/process` in local mode. OK.
3. **VRAM savings:** ~2-4 GB freed on Windows. This was the core motivation.
4. **`pip install .[local]`** no longer installs ollama. Users who need local LLM: `pip install .[local,local-llm]`.

### Tests
- Test `pip install .[local]` doesn't install ollama
- Test app starts without ollama package
- Test `/llm/local/status` returns clear error when ollama not installed

---

## Task 6: Config, Settings, Cache, and Format Validation

### What
Update all config layers. Fix format validation per provider. Add threshold field.

### `stt/config.py` — final
```python
class STTSettings(BaseSettings):
    mode: ProviderMode = ProviderMode.CLOUD

    # Cloud: Gemini (long audio / ai_prompt)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Cloud: Groq Whisper (short audio + normal)
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3-turbo"

    # Cloud routing threshold
    cloud_routing_threshold: float = 30.0  # seconds

    # Local: faster-whisper
    whisper_model_size: str = "large-v3-turbo"
    whisper_device: str = "auto"

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_STT_", env_file=".env", extra="ignore")

    @field_validator("cloud_routing_threshold")
    @classmethod
    def threshold_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cloud_routing_threshold must be > 0")
        return v
```

### `UserSettings` additions
```python
cloud_routing_threshold: float = 30.0
```

### `sync_to_runtime()` additions
```python
settings.stt.cloud_routing_threshold = us.cloud_routing_threshold
```

### `.env` template
```
# STT Cloud — Gemini (long audio)
JUSTSAY_STT_GEMINI_API_KEY=your-key-here
JUSTSAY_STT_GEMINI_MODEL=gemini-2.5-flash

# STT Cloud — Groq Whisper (short audio)
JUSTSAY_STT_GROQ_API_KEY=your-key-here
JUSTSAY_STT_GROQ_WHISPER_MODEL=whisper-large-v3-turbo

# Routing threshold (seconds)
JUSTSAY_STT_CLOUD_ROUTING_THRESHOLD=30.0
```

### Format validation per provider
Groq doesn't support `.webm`. Need to validate format matches provider capabilities.
```python
GROQ_SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}
GEMINI_SUPPORTED = {".wav", ".mp3", ".ogg", ".webm"}

# In routing: if file format not supported by selected provider, fallback to other
```

For live recording (always `.wav`): both providers support it, no issue.
For file uploads: check extension against provider's supported formats.

### API key documentation
In `.env` template, add comment:
```
# If you use the same Groq API key for both STT and LLM, set both:
# JUSTSAY_STT_GROQ_API_KEY=your-key
# JUSTSAY_LLM_GROQ_API_KEY=your-key
```
No code-level fallback. Explicit is better than implicit.

### Tests
- Test threshold validator rejects 0 and negative
- Test new config fields load from env
- Test UserSettings migration (old settings.json without new fields)
- Test sync_to_runtime syncs threshold
- Test format validation per provider

---

## Implementation Order
1. **Task 6** — Config changes (foundation for everything)
2. **Task 1** — Groq Whisper provider (new code, no breakage)
3. **Task 3** — Gemini style-aware prompts (modify existing, no breakage)
4. **Task 2** — Routing + cache refactor (connects everything)
5. **Task 4** — Remove LLM from pipeline (depends on 2)
6. **Task 5** — Remove Ollama from local deps (depends on 4)

## Risk Assessment
- **Low risk:** Tasks 1, 3, 5, 6 (additive or removal, no behavioral change)
- **Medium risk:** Tasks 2, 4 (changes core pipeline behavior)

## Rollback
- LLM code stays in repo. Revert `pipeline/service.py` to re-enable LLM step.
- Setting `threshold=0.001` makes almost everything go through Gemini (old-ish behavior without LLM).
- Standalone `/llm/process` endpoint always available for manual 2-step flow.

---

## QA & Architect Review Summary (Applied)

### CRITICAL fixes applied:
- [x] `recorder.duration_seconds` returns 0.0 after stop() — added `last_duration_seconds` property
- [x] Thread safety for dict cache — added `_cache_lock = threading.Lock()`

### HIGH fixes applied:
- [x] Pipeline no longer knows about specific providers — routing moved to `stt/__init__.py`
- [x] `transcribe_with_style()` removed — style via `**kwargs` or constructor, interface stays clean
- [x] `/process-file` missing `style` parameter — added
- [x] `_make_key` doesn't cover cloud provider params — replaced with dict[type] cache
- [x] Groq Whisper `cleanup()` defined

### MEDIUM fixes applied:
- [x] `style="ai_prompt"` always routes to Gemini, regardless of duration
- [x] `sync_to_runtime()` updated with new fields
- [x] Format validation per provider (`.webm` not supported by Groq)
- [x] Empty/refusal response handling for Gemini
- [x] Test migration plan for existing LLM-mocking tests
- [x] Concurrent request handling via thread-safe cache

### LOW acknowledged:
- `raw_text == cleaned_text` when LLM removed — documented as known limitation
- Logging format specified: `log.info("Pipeline route: %s, duration=%.1fs", ...)`
