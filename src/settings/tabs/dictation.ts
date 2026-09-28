import {
  acceleratorFromKeyEvent,
  detectShortcutPlatform,
  formatAccelerator,
  modifierHint,
} from "../../accelerator";
import { api, ApiRequestError, levelStream, type UserSettings } from "../../api";
import {
  EVENT_SETTINGS_CHANGED,
  EVENT_SHORTCUT_APPLIED,
  EVENT_SHORTCUT_REQUESTED,
  type ShortcutApplied,
  type ShortcutRequested,
} from "../../contracts";
import { saveSettings, cachePersistedShortcut, type TabLifecycle } from "../settings";
import { escapeHtml, meetingDisclosureHtml } from "../html";
import { isDecisiveRefusal, newSessionId } from "../../session";
import { notifyError } from "../../notify";
import { levelFromDb } from "../../level";
import { TimedOutError } from "../../timeout";
import { renderSelect } from "../../ui/controls";

/** A discard that could not be delivered states what is known and promises
 *  nothing: whether the device was released is exactly what this window cannot
 *  find out. That is true whatever failed it — `recorder.discard()` is reached
 *  before the handler can raise, so a 500, a timeout and a refused connection
 *  leave the device in the same unknown state.
 *
 *  The session id is kept when this label goes up, so the button stays on
 *  `Stop` and the next press sends the same discard again; so does closing the
 *  window, which now tears the tab down ([JS-121]). This window is the only one
 *  that can: the widget does not own this session and the whole point of the
 *  ownership guard is that it may not end what it does not own. The label must
 *  therefore not tell the user that pressing Stop closes anything — nothing
 *  here can establish that it did. */
const MICROPHONE_UNCONFIRMED_LABEL =
  "The backend did not answer — the microphone may still be open";

const LANGUAGES = [
  { code: "uk", label: "Ukrainian" },
  { code: "en", label: "English" },
  { code: "de", label: "German" },
  { code: "fr", label: "French" },
  { code: "es", label: "Spanish" },
  { code: "pl", label: "Polish" },
  { code: "ja", label: "Japanese" },
  { code: "zh", label: "Chinese" },
];

const RESULT_HINT = "setting-row-hint--result";
const SHORTCUT_HINT = "Hold it down while you speak";
const NO_MICROPHONE = "No microphone";
const UNKNOWN_MICROPHONE = "Your default microphone";

/** Dictation's everyday card — language, shortcut, microphone — with the
 *  meeting disclosure under it, added to the end of `container`. */
export function renderDictation(container: HTMLElement, settings: UserSettings): TabLifecycle {
  const platform = detectShortcutPlatform(navigator);
  let destroyed = false;

  container.insertAdjacentHTML(
    "beforeend",
    `
    <h2 class="panel-title">Dictation</h2>
    <p class="panel-subtitle">How talking turns into text.</p>
    <div class="card">
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Language</div>
          <div class="setting-row-hint">Pick the one you speak most</div>
        </div>
        <div class="setting-row-controls">
          <select id="lang-select" aria-label="Language">
            ${LANGUAGES.map(
              (l) => `<option value="${l.code}" ${l.code === settings.language ? "selected" : ""}>${l.label}</option>`
            ).join("")}
          </select>
        </div>
      </div>
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Shortcut</div>
          <div class="setting-row-hint" id="shortcut-hint" aria-live="polite">${SHORTCUT_HINT}</div>
        </div>
        <div class="setting-row-controls">
          <button type="button" class="keycap num" id="btn-shortcut" aria-describedby="shortcut-hint">${escapeHtml(formatAccelerator(settings.shortcut, platform))}</button>
        </div>
      </div>
      <div class="setting-row">
        <div class="setting-row-text">
          <div class="setting-row-title">Microphone</div>
          <div class="setting-row-hint" id="mic-hint" aria-live="polite">…</div>
        </div>
        <div class="setting-row-controls">
          <div class="level-meter"><i id="level-fill"></i></div>
          <button type="button" class="btn" id="btn-test-mic">Test</button>
        </div>
      </div>
    </div>
    <div class="legacy-tab">
      <div class="setting-group" id="meeting-consent-group">
        ${meetingDisclosureHtml(settings.meeting_consent_acknowledged)}
      </div>
    </div>
  `,
  );

  const langSelect = container.querySelector<HTMLSelectElement>("#lang-select")!;
  renderSelect(langSelect);
  langSelect.addEventListener("change", async () => {
    try {
      await saveSettings({ language: langSelect.value });
      await emitSettingsChanged();
    } catch (e) {
      notifyError(e instanceof Error ? e.message : String(e));
    }
  });

  const btnTest = container.querySelector<HTMLButtonElement>("#btn-test-mic")!;
  const micHint = container.querySelector<HTMLElement>("#mic-hint")!;
  const levelFill = container.querySelector<HTMLElement>("#level-fill")!;

  let micName = "…";
  let micHintShowsName = true;

  function showMicName() {
    micHintShowsName = true;
    micHint.textContent = micName;
    micHint.classList.remove(RESULT_HINT);
  }

  function showMicResult(text: string) {
    micHintShowsName = false;
    micHint.textContent = text;
    micHint.classList.add(RESULT_HINT);
  }

  void api.inputDevice().then(
    ({ name }) => {
      micName = name ?? NO_MICROPHONE;
      if (!destroyed && micHintShowsName) showMicName();
    },
    () => {
      micName = UNKNOWN_MICROPHONE;
      if (!destroyed && micHintShowsName) showMicName();
    },
  );

  let isRecording = false;
  /** The session this tab started and has not seen released. Kept across a
   *  failed discard on purpose: it is the only handle anything has on that
   *  capture, and dropping it would leave a microphone open that no surface in
   *  the app can reach. A teardown hands it to `sessionAwaitingRelease`, which
   *  outlives this closure, rather than dropping it. */
  let heldSession = "";
  let levelStreamAbort: AbortController | null = null;

  /** Both callbacks check that they are still the current stream before they
   *  write anything. A stream is detached in three places — a new one starting,
   *  the tab being destroyed, and a stop that failed — and in the last of those
   *  the label holds the only instruction the user has for closing a microphone
   *  that may still be open. A late error from a stream nobody is listening to
   *  must not overwrite it. */
  function startLevelStream(fill: HTMLElement) {
    stopLevelStream();
    const stream: AbortController = levelStream(
      (data) => {
        if (levelStreamAbort !== stream) return;
        fill.style.width = `${levelFromDb(data.level_db) * 100}%`;
      },
      () => {},
      (error) => {
        if (levelStreamAbort !== stream) return;
        showMicResult(`Recording — the level meter stopped: ${error}`);
      },
    );
    levelStreamAbort = stream;
  }

  function stopLevelStream() {
    if (levelStreamAbort) {
      levelStreamAbort.abort();
      levelStreamAbort = null;
    }
  }

  /** Whether the backend is holding *this* tab's capture.
   *
   *  Asked only after a start ran out of its budget — never after a refusal,
   *  which already answers the question and whose caller would otherwise wait
   *  a second 15 s budget before the label changed — and answered by the id
   *  rather than by `is_recording`: the recorder is process-wide, so a bare
   *  "something is recording" would let this window's Stop button end a
   *  dictation the user is in the middle of speaking. A read that fails proves
   *  nothing and is not adoption. */
  async function backendHoldsSession(session: string): Promise<boolean> {
    try {
      const status = await api.audioStatus();
      return status.is_recording && status.session_id === session;
    } catch {
      return false;
    }
  }

  function showRecording() {
    isRecording = true;
    btnTest.textContent = "Stop";
    showMicName();
    startLevelStream(levelFill);
  }

  /** The microphone test never wanted a file. `POST /audio/stop` wrote one WAV
   *  per press into the scratch directory whose size this same tab then
   *  displays, and only `/pipeline/dictate` ever deletes such a file, so every
   *  test left audio of the room on disk ([JS-122]). `POST /audio/discard`
   *  writes nothing, so there is nothing to delete and nothing to announce. */
  function showIdle() {
    heldSession = "";
    isRecording = false;
    btnTest.textContent = "Test";
    showMicName();
    stopLevelStream();
    levelFill.style.width = "0%";
  }

  /** A refusal the recorder produced from inside its own lock is an answer, not
   *  silence: `403` says another session holds the device and `409` says
   *  nothing is being recorded, and either way this tab's capture is closed.
   *  Reporting those as "the backend did not answer" was factually wrong and
   *  left the button on `Stop` with nothing that could ever clear it. */
  async function stopMicrophoneTest() {
    const session = heldSession;
    try {
      await api.audioDiscard(session);
    } catch (e) {
      console.error(e);
      if (destroyed) return;
      if (!isDecisiveRefusal(e)) {
        stopLevelStream();
        levelFill.style.width = "0%";
        showMicResult(MICROPHONE_UNCONFIRMED_LABEL);
        return;
      }
    }
    if (destroyed) return;
    showIdle();
  }

  /** There is no pre-flight `GET /audio/status` any more.
   *
   *  `POST /audio/start`'s own 409 is the authoritative answer to "is the
   *  microphone busy" and, unlike a separate read, it cannot be stale by the
   *  time it is acted on. The read that replaced it — the one above — asks a
   *  different question, about a request this tab already issued.
   *
   *  A start that runs out of its budget leaves the device open whichever way
   *  it ends, so the session is held from the moment the request goes out. The
   *  level stream is deliberately not started on the unconfirmed branch: it
   *  would report an error over the one instruction the user has for closing a
   *  microphone that may still be open. */
  async function startMicrophoneTest() {
    const session = newSessionId();
    heldSession = session;
    try {
      await api.audioStart(session);
    } catch (e) {
      console.error(e);
      if (destroyed) return;
      if (e instanceof TimedOutError) {
        const held = await backendHoldsSession(session);
        if (destroyed) return;
        if (held) {
          showRecording();
          return;
        }
        isRecording = true;
        btnTest.textContent = "Stop";
        showMicResult(MICROPHONE_UNCONFIRMED_LABEL);
        return;
      }
      heldSession = "";
      showMicResult(
        e instanceof ApiRequestError && e.status === 409
          ? "Microphone busy (widget recording)"
          : "Failed to start",
      );
      return;
    }
    if (destroyed) {
      releaseAndRemember(session);
      return;
    }
    showRecording();
  }

  btnTest.addEventListener("click", async () => {
    if (isRecording) {
      await stopMicrophoneTest();
    } else {
      await startMicrophoneTest();
    }
  });

  const shortcutBtn = container.querySelector<HTMLButtonElement>("#btn-shortcut")!;
  const shortcutHint = container.querySelector<HTMLElement>("#shortcut-hint")!;
  let recording = false;

  function showShortcutResult(text: string) {
    shortcutHint.textContent = text;
    shortcutHint.classList.add(RESULT_HINT);
  }

  if (sessionAwaitingRelease) releaseAndRemember(sessionAwaitingRelease);

  const consentGroup = container.querySelector<HTMLElement>("#meeting-consent-group")!;
  consentGroup
    .querySelector<HTMLButtonElement>("#btn-meeting-consent")!
    .addEventListener("click", async (event) => {
      const button = event.currentTarget as HTMLButtonElement;
      button.disabled = true;
      try {
        await saveSettings({ meeting_consent_acknowledged: true });
        if (destroyed) return;
        consentGroup.innerHTML = meetingDisclosureHtml(true);
      } catch (e) {
        if (destroyed) return;
        button.disabled = false;
        notifyError(e instanceof Error ? e.message : String(e));
      }
    });

  async function requestShortcut(shortcut: string, revertLabelTo: string) {
    try {
      const { emit } = await loadEventApi();
      const requested: ShortcutRequested = { shortcut };
      await emit(EVENT_SHORTCUT_REQUESTED, requested);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (!destroyed) {
        showShortcutResult(`Could not apply the shortcut: ${message}`);
        shortcutBtn.textContent = revertLabelTo;
      }
      notifyError(message);
    }
  }

  let captureHandler: ((e: KeyboardEvent) => void) | null = null;

  function stopCapture() {
    if (captureHandler) {
      document.removeEventListener("keydown", captureHandler, true);
      captureHandler = null;
    }
    recording = false;
    shortcutBtn.classList.remove("keycap--capturing");
  }

  shortcutBtn.addEventListener("click", () => {
    if (recording) return;
    recording = true;
    shortcutBtn.textContent = "Press keys…";
    shortcutBtn.classList.add("keycap--capturing");
    shortcutHint.textContent = SHORTCUT_HINT;
    shortcutHint.classList.remove(RESULT_HINT);

    captureHandler = (e: KeyboardEvent) => {
      if (destroyed) return;
      e.preventDefault();
      e.stopPropagation();

      const captured = acceleratorFromKeyEvent(e);
      if (!captured.ok) {
        if (captured.reason === "no-modifier") showShortcutResult(modifierHint(platform));
        return;
      }

      const activeLabel = formatAccelerator(settings.shortcut, platform);

      stopCapture();
      shortcutBtn.textContent = formatAccelerator(captured.accelerator, platform);
      showShortcutResult("Applying…");

      void requestShortcut(captured.accelerator, activeLabel);
    };

    document.addEventListener("keydown", captureHandler, true);
  });

  let unlistenShortcutApplied: (() => void) | null = null;
  void (async () => {
    try {
      const { listen } = await loadEventApi();
      const unlisten = await listen<ShortcutApplied>(EVENT_SHORTCUT_APPLIED, ({ payload }) => {
        if (destroyed) return;
        const label = formatAccelerator(payload.shortcut, platform);
        if (!payload.ok) {
          showShortcutResult(`${label} was not accepted: ${payload.reason ?? "unknown reason"}`);
          shortcutBtn.textContent = payload.stillActive
            ? formatAccelerator(payload.stillActive, platform)
            : "Not set";
          return;
        }
        if (payload.persisted === true) cachePersistedShortcut(payload.shortcut);
        showShortcutResult(
          payload.persisted === false
            ? `${label} is active now, but could not be saved: ${payload.reason ?? "unknown reason"}`
            : `${label} is active now.`,
        );
      });
      if (destroyed) unlisten();
      else unlistenShortcutApplied = unlisten;
    } catch {}
  })();

  /** The window was dismissed while this tab stays mounted, so everything that
   *  is not a held resource — the values on screen, the reads it has already
   *  paid for — is left exactly as it is. The discard follows the same path a
   *  press of `Stop` does, including the label it leaves when the backend does
   *  not answer. */
  function releaseResources() {
    if (heldSession) void stopMicrophoneTest();
  }

  return {
    destroy: () => {
      destroyed = true;
      stopCapture();
      stopLevelStream();
      if (heldSession) {
        releaseAndRemember(heldSession);
        heldSession = "";
      }
      if (unlistenShortcutApplied) {
        unlistenShortcutApplied();
        unlistenShortcutApplied = null;
      }
    },
    releaseResources,
  } satisfies TabLifecycle;
}

/** A capture whose release has not been confirmed, held outside any tab
 *  instance because the id is the only handle this app has on that device and
 *  a teardown would otherwise drop it. The next render of this tab retries it,
 *  which is what makes `heldSession`'s stated invariant true rather than
 *  aspirational. It is cleared only by an answer about that session: a 200, or
 *  a refusal the recorder gave from inside its own lock. */
let sessionAwaitingRelease = "";

function releaseAndRemember(sessionId: string): void {
  sessionAwaitingRelease = sessionId;
  void api.audioDiscard(sessionId).then(
    () => {
      if (sessionAwaitingRelease === sessionId) sessionAwaitingRelease = "";
    },
    (e) => {
      console.error(e);
      if (isDecisiveRefusal(e) && sessionAwaitingRelease === sessionId) sessionAwaitingRelease = "";
    },
  );
}

let eventApi: Promise<typeof import("@tauri-apps/api/event")> | null = null;

function loadEventApi(): Promise<typeof import("@tauri-apps/api/event")> {
  if (!eventApi) {
    eventApi = import("@tauri-apps/api/event").catch((e) => {
      eventApi = null;
      throw e;
    });
  }
  return eventApi;
}

export async function emitSettingsChanged() {
  try {
    const { emit } = await loadEventApi();
    await emit(EVENT_SETTINGS_CHANGED);
  } catch {
  }
}
