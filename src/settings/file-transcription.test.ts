// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EVENT_FILE_PICKED, MAX_UPLOAD_BYTES } from "../contracts";

const apiMock = {
  startFileJob: vi.fn(),
  jobs: vi.fn(),
};

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: apiMock,
}));

const { invokeMock, shellListeners, stopHearing } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
  shellListeners: new Map<string, (event: { payload: unknown }) => unknown>(),
  stopHearing: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, handler: (event: { payload: unknown }) => unknown) => {
    shellListeners.set(event, handler);
    return stopHearing;
  }),
}));

const {
  mountFileTranscription,
  refusalOf,
  ALREADY_TRANSCRIBING,
  ALREADY_UPLOADING,
  DROP_AREA_CLASS,
  DROP_HINT,
  REFUSAL_SHOWN_MS,
  UPLOAD_STOPPED,
} = await import("./file-transcription");
const { ApiRequestError, UPLOAD_STALL_MS } = await import("../api");
const { TimedOutError } = await import("../timeout");
type FileTranscription = import("./file-transcription").FileTranscription;
type UploadWatch = import("../api").UploadWatch;
type TranscriptionJob = import("../api").TranscriptionJob;

const onTaken = vi.fn();
const onSent = vi.fn(async () => {});
const idle: FileTranscription = {
  pick: () => {},
  destroy: () => {},
  uploads: { current: () => [], watch: () => () => {}, remove: () => {}, retry: () => {} },
};
let mounted: FileTranscription = idle;

/** The upload the API was last handed, with the watch it reports through. */
function lastSent(): { body: Blob; name: string; watch: UploadWatch } {
  const [body, name, watch] = apiMock.startFileJob.mock.lastCall!;
  return { body, name, watch };
}

/** A `startFileJob` that stays in flight until the test settles it or its signal aborts it. */
function heldUploads(): { answer: (id: string) => void; refuse: (err: unknown) => void }[] {
  const held: { answer: (id: string) => void; refuse: (err: unknown) => void }[] = [];
  apiMock.startFileJob.mockImplementation(
    (_body: Blob, _name: string, watch: UploadWatch) =>
      new Promise((resolve, reject) => {
        watch.signal.addEventListener("abort", () => reject(new DOMException("cancelled", "AbortError")));
        held.push({ answer: (id) => resolve({ id }), refuse: reject });
      }),
  );
  return held;
}

function uploads() {
  return mounted.uploads.current();
}

function buildFile(name: string, size: number): File {
  const file = new File(["audio-bytes"], name, { type: "audio/wav" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function drag(type: string, target: EventTarget, carried: string[], files: File[] = []): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", { value: { types: carried, files } });
  target.dispatchEvent(event);
  return event;
}

function picker(): HTMLInputElement {
  return document.querySelector<HTMLInputElement>('input[type="file"]')!;
}

function pick(file: File): void {
  Object.defineProperty(picker(), "files", { value: [file], configurable: true });
  picker().dispatchEvent(new Event("change"));
}

function overlay(): HTMLElement {
  return document.querySelector<HTMLElement>(".drop-overlay")!;
}

function shownText(): string {
  return overlay().querySelector(".drop-overlay-title")!.textContent ?? "";
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMock.startFileJob.mockResolvedValue({ id: "job-1" });
  apiMock.jobs.mockResolvedValue([]);
  document.body.innerHTML = `<main id="pane"><div id="child"><span id="grandchild"></span></div></main>`;
  mounted = mountFileTranscription(document.body, { onTaken, onSent });
});

afterEach(() => {
  mounted.destroy();
  vi.useRealTimers();
});

describe("refusalOf — why a file cannot be transcribed", () => {
  it("says a file of another kind is not audio", () => {
    expect(refusalOf(buildFile("notes.txt", 2048))).toBe("That's not an audio file");
    expect(refusalOf(buildFile("README", 2048))).toBe("That's not an audio file");
  });

  it("says an empty file is empty", () => {
    expect(refusalOf(buildFile("silence.wav", 0))).toBe("This file is empty");
  });

  it("names the ceiling a file is over", () => {
    expect(refusalOf(buildFile("long.mp3", MAX_UPLOAD_BYTES + 1))).toBe("This file is over 500 MB");
    expect(refusalOf(buildFile("edge.mp3", MAX_UPLOAD_BYTES))).toBeNull();
  });

  it("takes an audio file whatever the case of its extension", () => {
    expect(refusalOf(buildFile("Call.M4A", 2048))).toBeNull();
  });
});

describe("the overlay while a file is dragged over the window", () => {
  it("is hidden until a drag arrives", () => {
    expect(overlay().hidden).toBe(true);
  });

  it("dims the window under the drop hint when a file is dragged anywhere on it", () => {
    const enter = drag("dragenter", document.getElementById("grandchild")!, ["Files"]);

    expect(overlay().hidden).toBe(false);
    expect(shownText()).toBe("Drop to transcribe");
    expect(overlay().querySelector(".drop-overlay-hint")!.textContent).toBe(DROP_HINT);
    expect(DROP_HINT).toBe("mp3, wav, m4a, mp4 and 7 more · up to 500 MB");
    expect(enter.defaultPrevented, "the webview refuses a drag whose enter is not cancelled").toBe(true);
  });

  it("stays lit while the drag crosses from one element onto another", () => {
    const pane = document.getElementById("pane")!;
    const child = document.getElementById("child")!;

    drag("dragenter", pane, ["Files"]);
    drag("dragenter", child, ["Files"]);
    drag("dragleave", pane, ["Files"]);

    expect(overlay().hidden, "a leave fires at every element boundary; one leave must not hide it").toBe(false);

    drag("dragleave", child, ["Files"]);

    expect(overlay().hidden).toBe(true);
  });

  it("comes back when the drag leaves the window and enters again", () => {
    drag("dragenter", document.body, ["Files"]);
    drag("dragleave", document.body, ["Files"]);
    drag("dragleave", document.body, ["Files"]);
    expect(overlay().hidden).toBe(true);

    drag("dragenter", document.body, ["Files"]);

    expect(overlay().hidden, "an unmatched leave must not push the count below zero").toBe(false);
  });

  it("cancels dragover for a file, so the drop lands in the page", () => {
    drag("dragenter", document.body, ["Files"]);

    expect(drag("dragover", document.body, ["Files"]).defaultPrevented).toBe(true);
  });
});

describe("a drag that carries no file", () => {
  it("shows nothing and is left to the page, so a dragged path still lands in a field", () => {
    const enter = drag("dragenter", document.body, ["text/plain"]);
    const over = drag("dragover", document.body, ["text/plain"]);
    const drop = drag("drop", document.body, ["text/plain"]);

    expect(overlay().hidden).toBe(true);
    expect(enter.defaultPrevented).toBe(false);
    expect(over.defaultPrevented).toBe(false);
    expect(drop.defaultPrevented).toBe(false);
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });
});

describe("dropping a file", () => {
  it("sends it off as a job, hides the overlay and reports it sent", async () => {
    drag("dragenter", document.body, ["Files"]);
    const drop = drag("drop", document.getElementById("child")!, ["Files"], [buildFile("note.wav", 2048)]);

    expect(drop.defaultPrevented, "an unprevented drop navigates the webview to the file").toBe(true);
    expect(overlay().hidden).toBe(true);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(lastSent().name).toBe("note.wav");
    expect(uploads()).toEqual([]);
  });

  it("shows why a refused file was refused, without sending it, then goes away", async () => {
    vi.useFakeTimers();
    drag("dragenter", document.body, ["Files"]);
    drag("drop", document.body, ["Files"], [buildFile("notes.txt", 2048)]);

    await vi.waitFor(() => expect(overlay().hidden).toBe(false));
    expect(shownText()).toBe("That's not an audio file");
    expect(overlay().classList.contains("drop-overlay--refused")).toBe(true);
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
    expect(onTaken).not.toHaveBeenCalled();
    expect(uploads()).toEqual([]);

    vi.advanceTimersByTime(REFUSAL_SHOWN_MS);

    expect(overlay().hidden).toBe(true);
  });

  it("a click puts a refusal away at once", async () => {
    drag("drop", document.body, ["Files"], [buildFile("empty.wav", 0)]);
    await vi.waitFor(() => expect(shownText()).toBe("This file is empty"));

    overlay().click();

    expect(overlay().hidden).toBe(true);
  });

  it("a new drag after a refusal shows the drop hint again", async () => {
    drag("drop", document.body, ["Files"], [buildFile("big.wav", MAX_UPLOAD_BYTES + 1)]);
    await vi.waitFor(() => expect(shownText()).toBe("This file is over 500 MB"));

    drag("dragenter", document.body, ["Files"]);

    expect(shownText()).toBe("Drop to transcribe");
    expect(overlay().classList.contains("drop-overlay--refused")).toBe(false);
  });

  it("keeps the backend's reason on the file's card when it refuses the upload", async () => {
    apiMock.startFileJob.mockRejectedValue(new ApiRequestError("File too large (max 500MB)", 413));

    drag("drop", document.body, ["Files"], [buildFile("note.wav", 2048)]);

    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));
    expect(uploads()[0]).toMatchObject({ error: "File too large (max 500MB)", retryable: true });
    expect(overlay().hidden).toBe(true);
    expect(onSent).not.toHaveBeenCalled();
  });
});

describe("a file on its way to the backend", () => {
  it("shows at once, before any byte has gone, and brings the Files page up", () => {
    heldUploads();

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    expect(uploads()).toEqual([expect.objectContaining({ name: "interview.m4a", state: "uploading", sentFraction: 0 })]);
    expect(onTaken).toHaveBeenCalledOnce();
  });

  it("says how much of it has gone, and tells whoever watches", async () => {
    heldUploads();
    const seen = vi.fn();
    mounted.uploads.watch(seen);
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());
    seen.mockClear();

    lastSent().watch.onProgress(0.45);

    expect(uploads()[0].sentFraction).toBe(0.45);
    expect(seen).toHaveBeenCalledOnce();
  });

  it("sends a dropped file as it is, without reading it into memory first", async () => {
    const file = buildFile("interview.m4a", 2048);

    drag("drop", document.body, ["Files"], [file]);

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());
    expect(lastSent().body).toBe(file);
  });

  it("keeps its card until the job's card is drawn, so the two never both go missing", async () => {
    let drawn = () => {};
    onSent.mockImplementationOnce(() => new Promise<void>((resolve) => (drawn = resolve)));

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(uploads()).toHaveLength(1);

    drawn();

    await vi.waitFor(() => expect(uploads()).toEqual([]));
  });

  it("says the upload stopped when it timed out, and Try again sends the same file", async () => {
    const file = buildFile("interview.m4a", 2048);
    apiMock.startFileJob.mockRejectedValueOnce(new TimedOutError(UPLOAD_STALL_MS, "/jobs/file"));

    drag("drop", document.body, ["Files"], [file]);
    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));
    expect(uploads()[0]).toMatchObject({ error: UPLOAD_STOPPED, retryable: true });

    mounted.uploads.retry(uploads()[0]);

    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(apiMock.startFileJob).toHaveBeenCalledTimes(2);
    expect(lastSent().body).toBe(file);
    expect(uploads()).toEqual([]);
  });

  it("says the upload stopped when nothing answered at all", async () => {
    apiMock.startFileJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(uploads()[0]?.error).toBe(UPLOAD_STOPPED));
  });

  it("stops the upload when its card is closed, leaving no failure behind", async () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    mounted.uploads.remove(uploads()[0]);

    expect(lastSent().watch.signal.aborted).toBe(true);
    expect(uploads()).toEqual([]);
  });

  it("clears a failed card when it is closed", async () => {
    apiMock.startFileJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));

    mounted.uploads.remove(uploads()[0]);

    expect(uploads()).toEqual([]);
  });

  it("stops an upload in flight on teardown", async () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    mounted.destroy();
    mounted = idle;

    expect(lastSent().watch.signal.aborted).toBe(true);
  });
});

describe("several files at once", () => {
  it("takes every one of them and sends them one after another, in the order dropped", async () => {
    const held = heldUploads();

    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10), buildFile("b.mp3", 20), buildFile("c.ogg", 30)]);

    expect(uploads().map((u) => [u.name, u.state])).toEqual([
      ["a.m4a", "uploading"],
      ["b.mp3", "waiting"],
      ["c.ogg", "waiting"],
    ]);
    expect(onTaken).toHaveBeenCalledOnce();
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    held[0].answer("job-a");

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    expect(lastSent().name).toBe("b.mp3");
    expect(uploads().map((u) => [u.name, u.state])).toEqual([
      ["b.mp3", "uploading"],
      ["c.ogg", "waiting"],
    ]);
  });

  it("keeps a file dropped while another uploads waiting its turn", async () => {
    const held = heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    drag("drop", document.body, ["Files"], [buildFile("b.mp3", 20)]);
    await Promise.resolve();

    expect(apiMock.startFileJob).toHaveBeenCalledOnce();
    expect(uploads().map((u) => u.state)).toEqual(["uploading", "waiting"]);
    held[0].answer("job-a");
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
  });

  it("moves on to the next file when one fails", async () => {
    const held = heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10), buildFile("b.mp3", 20)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    held[0].refuse(new TypeError("Failed to fetch"));

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    expect(uploads().map((u) => [u.name, u.state])).toEqual([
      ["a.m4a", "failed"],
      ["b.mp3", "uploading"],
    ]);
  });

  it("moves on to the next file when the one uploading is closed", async () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10), buildFile("b.mp3", 20)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    mounted.uploads.remove(uploads()[0]);

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    expect(lastSent().name).toBe("b.mp3");
  });

  it("gives a refused one its own card with the reason, never sends it, and cannot retry it", async () => {
    heldUploads();

    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10), buildFile("notes.txt", 20)]);

    expect(uploads()[1]).toMatchObject({ name: "notes.txt", state: "failed", error: "That's not an audio file", retryable: false });
    expect(overlay().hidden).toBe(true);
    mounted.uploads.retry(uploads()[1]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());
    expect(lastSent().name).toBe("a.m4a");
    expect(uploads()[1].state).toBe("failed");
  });

  it("lets Choose a file pick several", () => {
    expect(picker().multiple).toBe(true);
  });
});

describe("dropping a file that is already on its way", () => {
  it("says it is already uploading and makes no second job", async () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledOnce());

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    expect(shownText()).toBe(ALREADY_UPLOADING);
    expect(overlay().classList.contains("drop-overlay--refused")).toBe(true);
    expect(uploads()).toHaveLength(1);
    expect(onTaken).toHaveBeenCalledOnce();
    expect(apiMock.startFileJob).toHaveBeenCalledOnce();
  });

  it("counts a file still waiting its turn as on its way", () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("a.m4a", 10), buildFile("b.mp3", 20)]);

    drag("drop", document.body, ["Files"], [buildFile("b.mp3", 20)]);

    expect(shownText()).toBe(ALREADY_UPLOADING);
    expect(uploads()).toHaveLength(2);
  });

  it("takes a different file of the same name", () => {
    heldUploads();
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 4096)]);

    expect(uploads()).toHaveLength(2);
    expect(overlay().hidden).toBe(true);
  });

  it("sends a failed file again in place of its failed card", async () => {
    apiMock.startFileJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(uploads()).toEqual([]);
  });
});

describe("picking a file with pick()", () => {
  it("opens the system file dialog, offering audio files", () => {
    const opened = vi.spyOn(picker(), "click").mockImplementation(() => {});

    mounted.pick();

    expect(opened).toHaveBeenCalledOnce();
    expect(picker().accept).toContain(".m4a");
  });

  it("sends the picked file off as a job and reports it sent", async () => {
    pick(buildFile("interview.m4a", 2048));

    expect(onTaken).toHaveBeenCalledOnce();
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(lastSent().name).toBe("interview.m4a");
    expect(overlay().hidden).toBe(true);
  });

  it("explains a picked file it refuses, the way it explains a dropped one", async () => {
    pick(buildFile("notes.txt", 2048));

    await vi.waitFor(() => expect(shownText()).toBe("That's not an audio file"));
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });
});

describe("a file picked from the ring or the tray", () => {
  async function shellPicks(name: string, size: number): Promise<void> {
    await vi.waitFor(() => expect(shellListeners.get(EVENT_FILE_PICKED)).toBeTypeOf("function"));
    shellListeners.get(EVENT_FILE_PICKED)!({ payload: { token: "token-1", name, size } });
  }

  it("fetches the bytes from the shell by token and sends them off as a job", async () => {
    invokeMock.mockResolvedValue(new Uint8Array([1, 2, 3, 4]).buffer);

    await shellPicks("Interview 3.m4a", 2048);

    expect(onTaken).toHaveBeenCalledOnce();
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(invokeMock).toHaveBeenCalledWith("take_picked_file", { token: "token-1" });
    expect(lastSent().name).toBe("Interview 3.m4a");
    expect(new Uint8Array(await lastSent().body.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  it("asks the shell for the bytes once, so Try again resends what it already read", async () => {
    invokeMock.mockResolvedValue(new ArrayBuffer(4));
    apiMock.startFileJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await shellPicks("Interview 3.m4a", 2048);
    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));
    const firstBody = lastSent().body;

    mounted.uploads.retry(uploads()[0]);

    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    expect(invokeMock).toHaveBeenCalledOnce();
    expect(lastSent().body).toBe(firstBody);
  });

  it("does not send a file whose card was closed while the shell was reading it", async () => {
    let read: (bytes: ArrayBuffer) => void = () => {};
    invokeMock.mockReturnValue(new Promise((resolve) => (read = resolve)));
    await shellPicks("Interview 3.m4a", 2048);
    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledOnce());

    mounted.uploads.remove(uploads()[0]);
    read(new ArrayBuffer(4));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(apiMock.startFileJob).not.toHaveBeenCalled();
    expect(uploads()).toEqual([]);
  });

  it("says the upload stopped when the shell never hands the bytes over", async () => {
    vi.useFakeTimers();
    invokeMock.mockReturnValue(new Promise(() => {}));
    await vi.advanceTimersByTimeAsync(0);
    shellListeners.get(EVENT_FILE_PICKED)!({ payload: { token: "token-1", name: "Interview 3.m4a", size: 2048 } });

    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS);

    expect(uploads()[0]).toMatchObject({ state: "failed", error: UPLOAD_STOPPED, retryable: true });
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("sends the bytes the shell handed over late when Try again is pressed, without asking twice", async () => {
    vi.useFakeTimers();
    let read: (bytes: ArrayBuffer) => void = () => {};
    invokeMock.mockReturnValue(new Promise((resolve) => (read = resolve)));
    await vi.advanceTimersByTimeAsync(0);
    shellListeners.get(EVENT_FILE_PICKED)!({ payload: { token: "token-1", name: "Interview 3.m4a", size: 2048 } });
    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS);
    expect(uploads()[0].state).toBe("failed");

    read(new ArrayBuffer(4));
    mounted.uploads.retry(uploads()[0]);
    await vi.advanceTimersByTimeAsync(0);

    expect(apiMock.startFileJob).toHaveBeenCalledOnce();
    expect(invokeMock).toHaveBeenCalledOnce();
  });

  it("refuses a file over the cap in the drop's words without reading it", async () => {
    await shellPicks("All hands.m4a", MAX_UPLOAD_BYTES + 1);

    await vi.waitFor(() => expect(shownText()).toBe("This file is over 500 MB"));
    expect(invokeMock).not.toHaveBeenCalled();
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
  });

  it("keeps the shell's reason on the card when the file cannot be read, with no Try again", async () => {
    invokeMock.mockRejectedValue("This file can't be read");

    await shellPicks("Interview 3.m4a", 2048);

    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));
    expect(uploads()[0]).toMatchObject({ error: "This file can't be read", retryable: false });
    expect(apiMock.startFileJob).not.toHaveBeenCalled();
    expect(onSent).not.toHaveBeenCalled();
  });

  it("stops hearing the shell on teardown", async () => {
    await vi.waitFor(() => expect(shellListeners.get(EVENT_FILE_PICKED)).toBeTypeOf("function"));

    mounted.destroy();
    mounted = idle;

    expect(stopHearing).toHaveBeenCalledOnce();
  });
});

describe("a drop area on the page", () => {
  function showDropArea(): void {
    document.getElementById("pane")!.innerHTML = `<div class="${DROP_AREA_CLASS}"></div>`;
  }

  const lit = () => document.body.classList.contains("dragging-file");

  it("lights the drop area instead of dimming the window when a file is dragged in", () => {
    showDropArea();

    const enter = drag("dragenter", document.getElementById("pane")!, ["Files"]);

    expect(lit()).toBe(true);
    expect(overlay().hidden).toBe(true);
    expect(enter.defaultPrevented).toBe(true);
  });

  it("puts the drop area back when the drag leaves the window", () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("dragleave", document.body, ["Files"]);

    expect(lit()).toBe(false);
  });

  it("starts the job of a file dropped anywhere and puts the drop area back", async () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("drop", document.getElementById("pane")!, ["Files"], [buildFile("note.wav", 2048)]);

    expect(lit()).toBe(false);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
  });

  it("still explains a refusal on the overlay", async () => {
    showDropArea();
    drag("dragenter", document.body, ["Files"]);

    drag("drop", document.body, ["Files"], [buildFile("notes.txt", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe("That's not an audio file"));
    expect(overlay().hidden).toBe(false);
  });

  it("takes a refusal away when the next drag arrives", async () => {
    showDropArea();
    drag("drop", document.body, ["Files"], [buildFile("empty.wav", 0)]);
    await vi.waitFor(() => expect(overlay().hidden).toBe(false));

    drag("dragenter", document.body, ["Files"]);

    expect(overlay().hidden).toBe(true);
    expect(lit()).toBe(true);
  });

  it("is not lit by a drag that carries no file", () => {
    showDropArea();

    drag("dragenter", document.body, ["text/plain"]);

    expect(lit()).toBe(false);
  });
});

describe("teardown", () => {
  it("removes the overlay and the dialog, and stops listening to the window", () => {
    mounted.destroy();
    mounted = idle;

    const enter = drag("dragenter", document.body, ["Files"]);

    expect(document.querySelector(".drop-overlay")).toBeNull();
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect(enter.defaultPrevented).toBe(false);
  });

  it("puts a lit drop area back", () => {
    document.getElementById("pane")!.innerHTML = `<div class="${DROP_AREA_CLASS}"></div>`;
    drag("dragenter", document.body, ["Files"]);

    mounted.destroy();
    mounted = idle;

    expect(document.body.classList.contains("dragging-file")).toBe(false);
  });
});

describe("dropping a file whose job is still on its way", () => {
  function fileJob(stage: TranscriptionJob["stage"]): TranscriptionJob {
    return { id: "job-1", kind: "file", name: "interview.m4a", stage, progress: null, error: null, entry_id: null };
  }

  async function sentOnce(): Promise<void> {
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(uploads()).toEqual([]));
    expect(onSent).toHaveBeenCalledOnce();
  }

  it("reads no jobs for a file when nothing was sent before it", async () => {
    await sentOnce();

    expect(apiMock.jobs).not.toHaveBeenCalled();
  });

  it.each(["queued", "transcribing", "saving"] as const)(
    "says it is already being transcribed while its job is %s, and makes no second job",
    async (stage) => {
      await sentOnce();
      apiMock.jobs.mockResolvedValue([fileJob(stage)]);

      drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

      await vi.waitFor(() => expect(shownText()).toBe(ALREADY_TRANSCRIBING));
      expect(overlay().classList.contains("drop-overlay--refused")).toBe(true);
      expect(apiMock.startFileJob).toHaveBeenCalledOnce();
      expect(uploads()).toEqual([]);
    },
  );

  it.each([
    ["done", [fileJob("done")]],
    ["failed", [fileJob("failed")]],
    ["gone from the list", []],
  ])("takes it again once its job is %s", async (_state, jobs) => {
    await sentOnce();
    apiMock.jobs.mockResolvedValue(jobs);

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    expect(overlay().hidden).toBe(true);
  });

  it("counts it as being transcribed when the jobs cannot be read, rather than send it twice", async () => {
    await sentOnce();
    apiMock.jobs.mockRejectedValue(new TypeError("Failed to fetch"));

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe(ALREADY_TRANSCRIBING));
    expect(apiMock.startFileJob).toHaveBeenCalledOnce();
  });

  it("takes a different file of the same name", async () => {
    await sentOnce();
    apiMock.jobs.mockResolvedValue([fileJob("queued")]);

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 4096)]);

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
  });

  it("takes the others of several files and says it skipped the one being transcribed", async () => {
    await sentOnce();
    apiMock.jobs.mockResolvedValue([fileJob("transcribing")]);

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048), buildFile("b.mp3", 20)]);

    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    expect(lastSent().name).toBe("b.mp3");
    expect(shownText()).toBe(ALREADY_TRANSCRIBING);
  });

  it("says it is being transcribed, not uploading, once the backend has queued it", async () => {
    let drawn = () => {};
    onSent.mockImplementationOnce(() => new Promise<void>((resolve) => (drawn = resolve)));
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    apiMock.jobs.mockResolvedValue([fileJob("queued")]);

    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe(ALREADY_TRANSCRIBING));
    drawn();
  });

  it("keeps a file sent while the jobs were being read, so it is not sent twice", async () => {
    apiMock.startFileJob.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    drag("drop", document.body, ["Files"], [buildFile("late.wav", 30)]);
    await vi.waitFor(() => expect(uploads()[0]?.state).toBe("failed"));
    apiMock.startFileJob.mockResolvedValueOnce({ id: "job-a" });
    drag("drop", document.body, ["Files"], [buildFile("first.wav", 10)]);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledOnce());
    let answerRead: (jobs: TranscriptionJob[]) => void = () => {};
    apiMock.jobs.mockReturnValueOnce(new Promise((resolve) => (answerRead = resolve)));
    drag("drop", document.body, ["Files"], [buildFile("other.wav", 20)]);
    await vi.waitFor(() => expect(apiMock.jobs).toHaveBeenCalledOnce());

    apiMock.startFileJob.mockResolvedValueOnce({ id: "job-late" });
    mounted.uploads.retry(uploads().find((u) => u.name === "late.wav")!);
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledTimes(2));
    answerRead([]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(4));
    apiMock.jobs.mockResolvedValue([{ ...fileJob("queued"), id: "job-late", name: "late.wav" }]);

    drag("drop", document.body, ["Files"], [buildFile("late.wav", 30)]);

    await vi.waitFor(() => expect(shownText()).toBe(ALREADY_TRANSCRIBING));
    expect(apiMock.startFileJob).toHaveBeenCalledTimes(4);
  });

  it("does not let a slow read of the old job forget the file's newer one", async () => {
    await sentOnce();
    let answerSlow: (jobs: TranscriptionJob[]) => void = () => {};
    apiMock.jobs.mockReturnValueOnce(new Promise((resolve) => (answerSlow = resolve)));
    drag("drop", document.body, ["Files"], [buildFile("other.wav", 20)]);
    apiMock.jobs.mockResolvedValueOnce([]);
    apiMock.startFileJob.mockResolvedValueOnce({ id: "job-2" });
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(onSent).toHaveBeenCalledTimes(2));

    answerSlow([]);
    await vi.waitFor(() => expect(apiMock.startFileJob).toHaveBeenCalledTimes(3));
    apiMock.jobs.mockResolvedValue([{ ...fileJob("transcribing"), id: "job-2" }]);
    drag("drop", document.body, ["Files"], [buildFile("interview.m4a", 2048)]);

    await vi.waitFor(() => expect(shownText()).toBe(ALREADY_TRANSCRIBING));
    expect(apiMock.startFileJob).toHaveBeenCalledTimes(3);
  });
});
